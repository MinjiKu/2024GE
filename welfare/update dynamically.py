import sys
import os

# Get the absolute path of the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(script_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)
import var
# # to check if actual tariffs get calculated
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Times New Roman 폰트를 사용하도록 설정
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] + rcParams['font.serif']

# var.fill_gamma()
# var.fill_pi()
factual_pi = var.pi.copy() #factual var.pi 보존

# var.fill_alpha()
def update_economic_variables(tau, j):
    # Update t based on the new tau values
    for i in tau.keys():
            if i==j: continue
            for industry in var.industries:
                    var.t[i][j][industry] = max(tau[i][j][industry] - 100, 1e-10)

    # Recalculate gamma, pi, and alpha based on the updated t values
    var.fill_gamma()
    var.fill_alpha()
    var.fill_pi()

    # Update p_ijs and T based on the new values of t
    for i in var.countries:
        for j in var.countries:
            if i != j:
                for industry in var.industries:
                    change_rate = (var.tau[i][j][industry] - previous_tau[i][j][industry]) / previous_tau[i][j][industry] if previous_tau[i][j][industry] != 0 else 0
                    var.p_ijs[i][j][industry] *= (1 + change_rate)
                    var.T[i][j][industry] *= (1 - change_rate * var.de[industry])
def calc_x(j, s):
    sum = 0
    TR = 0
    for i in var.countries:
        if i == j: continue
        TR += var.t[i][j][s] * var.T[i][j][s]
    
    sum = var.w[j] * var.L_js[j][s] + var.pi[j][s] + var.L_js[j][s]/var.L_j[j] * TR
    
    return sum

# Welfare function
def calc_welfare(j, s):
    return calc_x(j, s) / var.P_j[j]

def gov_obj(tau, j):
    # Update economic variables based on the current tau_js
    update_economic_variables(tau, j)

    total = 0
    for s in var.industries:
        total += var.pol_econ[j][s] * calc_welfare(j, s)

    return -total  # We minimize, so we return the negative

def eq_12(j, s):
    total = 0
    for i in var.countries:
        if i != j:
            total += (var.gamma[i][j][s] * (var.tau_hat[i][j][s] ** (1 - var.sigma[s]))) ** (1 / (1 - var.sigma[s]))
    var.P_hat[j][s] = total
    return var.P_hat[j][s] - total

def x2(j):
    total = 0
    for i in var.countries:
        for s in var.industries:
            if i != j:
                total += var.t[i][j][s] * var.T[i][j][s]
    return total

def wL(j):
    term2 = 0
    for i in var.countries:
        for s in var.industries:
            if i != j:
                term2 += var.t[i][j][s] * var.T[i][j][s]

    term3 = 0
    for s in var.industries:
        term3 += var.pi[j][s]

    return x2(j) - term2 - term3

def complicated(j):
    total = 0
    for i in var.countries:
        for s in var.industries:
            if i != j:
                total += (var.t[i][j][s] * var.T[i][j][s] / x2(j) * var.t_hat[i][j][s] * 
                        (var.P_hat[j][s]** (var.sigma[s] - 1)) * (abs(var.tau_hat[i][j][s]) ** -var.sigma[s]) + (var.pi[j][s] / x2(j) * var.pi_hat[j][s]))
    return total

def term3(j):
    total = 0
    for s in var.industries:
        for i in var.countries:
            if i != j:
                total += (var.pi[j][s] / x2(j) * var.pi_hat[j][s]) * var.alpha[j][i][s] * (abs(var.tau_hat[j][i][s]) ** -var.sigma[s]) * (var.w[i] ** (1 - var.sigma[s])) * (var.P_hat[j][s] ** (var.sigma[s] - 1))
    return total

def eq_13(j):
    epsilon = 1e-10
    term1 = wL(j) / (x2(j) + epsilon)
    term2 = complicated(j)
    term3 = 0

    aggregated_x = 0

    for s in var.industries:
        term3 += var.pi[j][s] / (x2(j) + epsilon) * var.pi_hat[j][s]
        aggregated_x += calc_x(j, s)
    
    sum = (term1) / (1 - term2 - term3 + epsilon)

    return term1 + term2 + term3 - aggregated_x

def eq_10(i, s):
    total = 0
    for j in var.countries:
        if i != j:
            total += (var.alpha[i][j][s] * (var.tau_hat[i][j][s] ** -var.sigma[s]) * 
                      (var.w[i] ** (1 - var.sigma[s])) * (var.P_hat[j][s] ** (var.sigma[s] - 1)) * var.X_hat[j])
    
    return total - var.pi_hat[i][s]

def global_constraints_generator(tau):
    cons = []
    
    # Constraint 1: eq_12 for every country j and industry s
    for j in var.countries:
        for s in var.industries:
            cons.append({'type': 'eq', 'fun': lambda tau, j=j, s=s: eq_12(j, s)})
    
    # Constraint 2: eq_13 for every country j
    for j in var.countries:
        cons.append({'type': 'eq', 'fun': lambda tau, j=j: eq_13(j)})
    
    # Constraint 3: eq_10 for every country i and industry s
    for i in var.countries:
        for s in var.industries:
            cons.append({'type': 'eq', 'fun': lambda tau, i=i, s=s: eq_10(i, s)})
    
    return cons

def flatten_dict(tau_dict):
    flat_list = []
    for i in tau_dict.keys():  # Iterate over all exporter keys
        for industry in var.industries:  # Iterate over all industries
            flat_list.append(tau_dict[i][industry])
    return flat_list

def unflatten_dict(flat_list, j):
    unflattened_dict = {}
    index = 0
    
    for i in var.countries:  # Iterate over all exporter keys
        if i != j:  # Skip the importer itself
            unflattened_dict[i] = {j: {}}  # Initialize the structure
            for industry in var.industries:  # Iterate over all industries
                unflattened_dict[i][j][industry] = flat_list[index]
                index += 1
    
    return unflattened_dict

def welfare_change():
    delta_W_W = {}

    x_js_list = {country: {industry: 0 for industry in var.industries} for country in var.countries}
    x_j_list = {country: 0 for country in var.countries}
    
    for j in var.countries:
        for s in var.industries:
            x_js_list[j][s] = calc_x(j, s)
            x_j_list[j] += x_js_list[j][s]
   
    
    for j in var.countries:  # j국 (수입국)
        term1 = 0
        term2 = 0
        term3 = 0
        
        for i in var.countries:
            if i != j:
                for s in var.industries: 
                    term1 += (var.T[i][j][s] / x_j_list[j]) * (changerate_p_js[j][s] - changerate_p_js[i][s])
                    term3 += (var.t[i][j][s] * var.T[i][j][s] / x_j_list[j]) * (changerate_T[i][j][s]-changerate_p_is[i][s])
        
        for s in var.industries:  # s산업
            term2 += (var.pi[j][s] / x_j_list[j]) * (changerate_pi[j][s] - changerate_p_js[j][s])
        
        delta_W_W[j] = term1 + term2 + term3
    
    return delta_W_W

p_is = {i: {s: 0 for s in var.industries} for i in var.countries}
p_js = {j: {s: float('inf') for s in var.industries} for j in var.countries}

# Ensure the directory exists
output_dir = "nash_img"
os.makedirs(output_dir, exist_ok=True)

# Update hats function
def update_hats(tau, t, pi): #갱신된 값이 인자로 들어감
    # global pi_hat, tau_hat, t_hat, factual_tau, factual_pi, factual_t
    global factual_pi
    for i in var.countries:
        for j in var.countries:
            if i != j:
                for s in var.industries:
                    var.tau_hat[i][j][s] = abs(tau[i][j][s] / var.factual_tau[i][j][s])
                    var.t_hat[i][j][s] = abs(t[i][j][s] / var.factual_t[i][j][s])
    for j in var.countries:
        for s in var.industries:
            var.pi_hat[j][s] = abs(var.pi[j][s] / factual_pi[j][s])


def calculate_p_js(p_ijs):
    p_js = {i: {s: 1e-10 for s in var.industries} for i in var.countries}
    
    for i in var.countries:
        for s in var.industries:
            max_value = float('-inf')
            for j in var.countries:
                if j != i:
                    max_value = max(max_value, p_ijs[i][j][s])
            p_js[i][s] = max_value
    
    return p_js

def calculate_p_is(p_ijs):
    p_is = {j: {s: float('inf') for s in var.industries} for j in var.countries}
    
    for j in var.countries:
        for s in var.industries:
            min_value = float('inf')
            for i in var.countries:
                if i != j:
                    current_value = p_ijs[i][j][s]
                    if current_value == 0:
                        current_value = 1e-10  # Prevent zero values
                    min_value = min(min_value, current_value)
            p_is[j][s] = max(min_value, 1e-10)  # Ensure min_value isn't zero
    
    return p_is

def optimize_for_importer(j):
    # Flatten the tau structure for the current importer
    initial_tau = flatten_dict({
        i: {s: var.tau[i][j][s] for s in var.industries}
        for i in var.countries if i != j
    })
    #print("initial_tau for ",j, initial_tau)
    
    # Define the local objective function for the importer
    def importer_gov_obj(flat_tau):
        unflattened_tau = unflatten_dict(flat_tau, j)
        update_economic_variables(unflattened_tau, j)
        return gov_obj(unflattened_tau, j)

    # Perform the optimization for the specific importer
    result = minimize(
        importer_gov_obj,          # Local objective function
        initial_tau,               # Initial guess for tau
        method='L-BFGS-B',         # Optimization method
        tol=1e-12,                 # Tolerance for termination (set very small to ensure high precision)
        options={
            'disp': True,         # Display convergence messages
            'maxiter': 10000,     # Maximum number of iterations (set high if necessary)
            'ftol': 1e-12,        # Precision goal for the value of the function being optimized
            'gtol': 1e-12,        # Precision goal for the gradient of the function being optimized
        }
    )

    # Map the results back to the original tau structure
    optimized_tau = unflatten_dict(result.x, j)
    optimized_g = -result.fun

    print("importer gov obj result\n",
          "optimized_tau:", optimized_tau,
          "\noptimized_g:", optimized_g)
    
    return optimized_tau, optimized_g

tariff_history = {i: {j: {industry: [] for industry in var.industries} for j in var.countries if j != i} for i in var.countries}
welfare_history = {j: [] for j in var.countries} 

iteration = 50
# Perform 100 iterations
for iter in range(iteration):
    print(f"Iteration {iter + 1}")
    print(var.tau)

    previous_tau = var.tau.copy()
    # Iterate over each importing country and optimize
    for j in var.countries:
        print(f"Optimizing tariffs for imports into country: {j}")
        optimized_tau, optimized_g = optimize_for_importer(j)

        #print(f"Welfare for {j} after optimization: {optimized_g}")
        welfare_history[j].append(optimized_g)
        #print(f"print welfare history for {j}: {welfare_history[j]}")
        for i in var.countries:
            if i != j:
                for s in var.industries:
                    var.tau[i][j][s] = optimized_tau[i][j][s]
                    tariff_history[i][j][s].append(max(optimized_tau[i][j][s], 1e-10)) 
    
    print(f"tariff history after {iter+1} optimization: ", tariff_history)
    print(f"welfare history after {iter+1} optimization: ", welfare_history)
    # Store the current state of the economic variables
    temp_p = var.p_ijs.copy() 
    temp_T = var.T.copy()
    temp_pi = var.pi.copy()
    temp_p_is = calculate_p_is(var.p_ijs)
    temp_p_js = calculate_p_js(var.p_ijs)

    # Update economic variables with the new tau after all optimizations
    update_economic_variables(var.tau, j)

    # Recalculate p_is and p_js after updating tau
    p_is = calculate_p_is(var.p_ijs)
    p_js = calculate_p_js(var.p_ijs)

    #print("p_js for iteration:", iter+1, p_js)

    var.fill_pi()
    var.fill_gamma()
    var.fill_alpha()

    # Recalculate gamma, var.pi, and alpha with new tau values
    update_hats(var.tau, var.t, var.pi)

    #Delta 값 계산
    changerate_pi = {country: {industry: (var.pi[country][industry] - temp_pi[country][industry])/temp_pi[country][industry] for industry in var.industries} for country in var.countries}
    changerate_p_is = {i: {industry: (p_is[i][industry] - temp_p_is[i][industry])/ temp_p_is[i][industry] for industry in var.industries} for i in var.countries}
    changerate_p_js = {j: {industry: (p_js[j][industry] - temp_p_js[j][industry])/temp_p_js[j][industry] for industry in var.industries} for j in var.countries}
    changerate_T = {i: {j: {industry: (var.T[i][j][industry] - temp_T[i][j][industry])/temp_T[i][j][industry] for industry in var.industries} for j in var.countries if i != j} for i in var.countries}
        # Call welfare_change with updated delta values
    print("welfare change effect of iteration", (iter+1), welfare_change())
    print("\nCorresponding t values:")
    for i in var.countries:
        print(f"\nt values for {i} as the home country:")
        df_t = pd.DataFrame({j: {s: var.t[i][j][s] for s in var.industries} for j in var.countries if j != i})
        print(df_t)

    # Print the current state of var.tau
    print("Nash Tariffs (tau) after iteration", iter + 1)
    for i in var.countries:
        print(f"\nTariffs for {i} as the home country:")
        df_tau = pd.DataFrame({j: {s: var.tau[i][j][s] for s in var.industries} for j in var.countries if j != i})
        print(df_tau)

iter_list = list(range(1, iteration + 1))
for exporter in var.countries:
    for importer in var.countries:
        if exporter != importer:
            for industry in var.industries:
                tariffs = tariff_history[exporter][importer][industry]

                # Calculate the differences between consecutive tariffs
                tariff_diffs = np.diff(tariffs)

                # Optional: amplify the differences
                amplified_diffs = tariff_diffs * 1e6

                plt.figure(figsize=(10, 6))
                plt.plot(iter_list[1:], amplified_diffs, marker='o', color='#bb0a1e', linewidth=2)
                
                # Optional: set the y-limits manually to zoom in
                plt.ylim([np.min(amplified_diffs) - 0.01, np.max(amplified_diffs) + 0.01])
                
                plt.title(f'Change in Tariff for "{industry}" from {exporter} to {importer} in Repeated Game')
                plt.xlabel('Iteration')
                plt.ylabel('Amplified Change in Tariff')
                plt.grid(True)

                # Save the plot
                file_name = f"{output_dir}/change_tariff_{industry}_{exporter}_to_{importer}.png"
                plt.savefig(file_name)
                plt.close()

for exporter in var.countries:
    for importer in var.countries:
        if exporter != importer:
            for industry in var.industries:
                tariffs = tariff_history[exporter][importer][industry]

                # Calculate the percentage change between consecutive tariffs
                tariff_changes = np.diff(tariffs) / tariffs[:-1] * 100

                plt.figure(figsize=(10, 6))
                plt.plot(iter_list[1:], tariff_changes, marker='o', color='#bb0a1e', linewidth=2)

                # Optional: set the y-limits manually to zoom in
                plt.ylim([np.min(tariff_changes) - 0.01, np.max(tariff_changes) + 0.01])

                plt.title(f'Change Rate in Tariff for "{industry}" from {exporter} to {importer} in Repeated Game')
                plt.xlabel('Iteration')
                plt.ylabel('Change Rate in Tariff (%)')
                plt.grid(True)

                # Save the plot
                file_name = f"{output_dir}/change_rate_tariff_{industry}_{exporter}_to_{importer}.png"
                plt.savefig(file_name)
                plt.close()

for importer in var.countries:
    welfares = welfare_history[importer]

    plt.figure(figsize=(10, 6))
    plt.plot(iter_list, welfares, marker='o', color='#bb0a1e', linewidth = 2)
    #plt.ylim([1.0, 1.5])
    plt.grid(True, linestyle='--', alpha=0.7)


    # for i, txt in enumerate(welfares):
    #     if i == 0 or txt != welfares[i-1]:  # 첫 포인트이거나, 앞의 값과 다를 때만 표시
    #         plt.annotate(f'{txt:.1f}', (iter_list[i], welfares[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title(f'Welfare for {importer} in Repeated Game')
    plt.xlabel('Reapeated Game')
    plt.ylabel('Welfare')
    plt.grid(True)
    
    # Save the plot
    file_name = f"{output_dir}/welfare_{importer}.png"
    plt.savefig(file_name)
    plt.close()

for importer in var.countries:
    welfares = welfare_history[importer]

    # Calculate the percentage changes between consecutive welfare values
    percentage_changes = [(welfares[i] - welfares[i-1]) / welfares[i-1] * 100 for i in range(1, len(welfares))]

    plt.figure(figsize=(10, 6))
    plt.plot(iter_list[1:], percentage_changes, marker='o', color='#bb0a1e', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Annotate the plot with the percentage change values
    for i, txt in enumerate(percentage_changes):
        plt.annotate(f'{txt:.2f}%', (iter_list[i+1], percentage_changes[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title(f'Welfare Change Rate for {importer} in Repeated Game')
    plt.xlabel('Iteration')
    plt.ylabel('Percentage Change in Welfare')
    plt.grid(True)
    
    # Save the plot
    file_name = f"{output_dir}/welfare_change_rate_{importer}.png"
    plt.savefig(file_name)
    plt.close()

#%%

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

var.fill_gamma()
var.fill_pi()
factual_pi = var.pi.copy() #factual var.pi 보존


var.fill_alpha()
def update_economic_variables(tau):
    # Update t based on the new tau values
    for i in var.countries:
        for j in var.countries:
            if i != j:
                for industry in var.industries:
                    var.t[i][j][industry] = max(tau[i][j][industry] - 1, 1e-10)

    # Recalculate gamma, pi, and alpha based on the updated t values
    var.fill_gamma()
    var.fill_alpha()
    var.fill_pi()

    # Update p_ijs and T based on the new values of t
    for i in var.countries:
        for j in var.countries:
            if i != j:
                for industry in var.industries:
                    previous_tau = var.factual_tau[i][j][industry]
                    change_rate = (var.tau[i][j][industry] - previous_tau) / previous_tau if previous_tau != 0 else 0
                    var.p_ijs[i][j][industry] *= (1 + change_rate)
                    var.T[i][j][industry] *= (1 + change_rate * var.de[industry])
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
    update_economic_variables(tau)

    total = 0
    for s in var.industries:
        total += var.pol_econ[j][s] * calc_welfare(j, s)

    print("total (gov_obj) for",j,iter,":",total)

    return -total  # We minimize, so we return the negative

# Constraint 1 for country j and industry s
def eq_12(j, s):
    total = 0
    for i in var.countries:
        if i != j:
            total += (var.gamma[i][j][s] * (var.tau_hat[i][j][s] ** (1 - var.sigma[s]))) ** (1 / (1 - var.sigma[s]))
    var.P_hat[j][s] = total
    return var.P_hat[j][s] - total

# Constraint 2 helper functions
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

# Constraint 3 for country i and industry s
def eq_10(i, s):
    total = 0
    for j in var.countries:
        if i != j:
            total += (var.alpha[i][j][s] * (var.tau_hat[i][j][s] ** -var.sigma[s]) * 
                      (var.w[i] ** (1 - var.sigma[s])) * (var.P_hat[j][s] ** (var.sigma[s] - 1)) * var.X_hat[j])
    
    return total - var.pi_hat[i][s]

def global_constraints_generator(tau_js):
    cons = []
    
    # Constraint 1: eq_12 for every country j and industry s
    for j in var.countries:
        for s in var.industries:
            cons.append({'type': 'eq', 'fun': lambda tau_js, j=j, s=s: eq_12(j, s)})
    
    # Constraint 2: eq_13 for every country j
    for j in var.countries:
        cons.append({'type': 'eq', 'fun': lambda tau_js, j=j: eq_13(j)})
    
    # Constraint 3: eq_10 for every country i and industry s
    for i in var.countries:
        for s in var.industries:
            cons.append({'type': 'eq', 'fun': lambda tau_js, i=i, s=s: eq_10(i, s)})
    
    return cons

def flatten_dict(tau_dict):
    flat_list = []
    for i in var.countries:
        for j in var.countries:
            if i != j:
                for industry in var.industries:
                    flat_list.append(tau_dict[i][j][industry])
    return flat_list

def unflatten_dict(flat_list):
    tau_dict = {i: {j: {industry: 0 for industry in var.industries} for j in var.countries if j != i} for i in var.countries}
    idx = 0
    for i in var.countries:
        for j in var.countries:
            if i != j:
                for industry in var.industries:
                    tau_dict[i][j][industry] = flat_list[idx]
                    idx += 1
    return tau_dict

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
        
        for i in var.countries:  # i국 (수출국)
            if i != j:
                for s in var.industries:  # s산업
                    #print(f'{j}{cal_x_j(j)}')
                    term1 += (var.T[i][j][s] / x_j_list[j]) * (changerate_p_js[j][s] - changerate_p_js[i][s])
                    term3 += (var.t[i][j][s] * var.T[i][j][s] / x_j_list[j]) * (changerate_T[i][j][s]-changerate_p_is[i][s])
        
        for s in var.industries:  # s산업
            term2 += (var.pi[j][s] / x_j_list[j]) * (changerate_pi[j][s] - changerate_p_js[j][s])
        
        delta_W_W[j] = term1 + term2 + term3
    
    return delta_W_W

# Initialize a dictionary to store tariff values for each iteration
tariff_history = {i: {j: {industry: [] for industry in var.industries} for j in var.countries if j != i} for i in var.countries}
welfare_history = {i: {j: {industry: [] for industry in var.industries} for j in var.countries if j != i} for i in var.countries}
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


def calculate_optimum_tariffs():
    global tau, t, tariff_matrices

    # Initialize dictionaries for storing results
    optimal_taus = {i: {j: {industry: var.tau[i][j][industry] for industry in var.industries} for j in var.countries if j != i} for i in var.countries}
    gov_obj_values = {i: {j: {industry: 0 for industry in var.industries} for j in var.countries if j != i} for i in var.countries}

    # Flatten the tau structure to create an initial guess for the optimizer
    initial_tau = flatten_dict(optimal_taus)

    # Define the global objective function that operates on the entire tau structure
    def global_gov_obj(flat_tau):
        total_obj = 0
        unflattened_tau = unflatten_dict(flat_tau)  # This reconstructs the tau structure
        update_economic_variables(unflattened_tau)  # Update t, T, pi, etc. based on the new tau
        
        for importer in var.countries:
            total_obj += gov_obj(unflattened_tau, importer)
        return total_obj

    # Generate global constraints for all countries and industries
    global_constraints = global_constraints_generator(initial_tau)

    # Perform global minimization across all countries and industries
    result = minimize(global_gov_obj, initial_tau, constraints=global_constraints)
    
    # Map the results back into the optimal_taus structure
    unflattened_result_tau = unflatten_dict(result.x)
    for i in var.countries:
        for j in var.countries:
            if i != j:
                for industry in var.industries:
                    optimal_taus[i][j][industry] = unflattened_result_tau[i][j][industry]
                    gov_obj_values[i][j][industry] = -result.fun
    
    return optimal_taus, gov_obj_values


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


iteration = 20
# Perform 100 iterations
for iter in range(iteration):
    print(f"Iteration {iter + 1}")
    print(var.tau)
    new_taus = {i: {j: {industry: 0 for industry in var.industries} for j in var.countries if j != i} for i in var.countries}
    gov_obj_values = {i: {j: {industry: 0 for industry in var.industries} for j in var.countries if j != i} for i in var.countries}
    
    new_taus, gov_obj_values = calculate_optimum_tariffs()

    #delta값 계산을 위해 기존 값 저장
    temp_p = var.p_ijs.copy() 
    temp_T = var.T.copy()
    temp_pi = var.pi.copy()
    temp_p_is = calculate_p_is(var.p_ijs)
    temp_p_js = calculate_p_js(var.p_ijs)

    # Apply new_tau and recalculate the dependent variables
    update_economic_variables(new_taus)

    for i in var.countries:
        for j in var.countries:
            if j != i:
                for industry in var.industries:
                    tariff_history[i][j][industry].append(var.tau[i][j][industry])  # Store the tariff value
                    welfare_history[i][j][industry].append(gov_obj_values[i][j][industry])
    
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
    
# Print the final Nash tariffs and corresponding t values
    print("Nash Tariffs (tau):")
    for i in var.countries:
        print(f"\nTariffs for {i} as country_i:")
        df_tau = pd.DataFrame({j: {s: var.tau[i][j][s] for s in var.industries} for j in var.countries if j != i})
        print(df_tau)

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

    # Recalculate gamma, var.pi, and alpha with new tau values
    update_hats(var.tau, var.t, var.pi)

iter_list = list(range(1,iteration+1))
for exporter in var.countries:
    for importer in var.countries:
        if exporter != importer:
            for industry in var.industries:
                tariffs = tariff_history[exporter][importer][industry]

                plt.figure(figsize=(10, 6))
                plt.plot(iter_list, tariffs, marker='o', color='#bb0a1e', linewidth = 2)
                plt.ylim([1.0, 1.5])

                for i, txt in enumerate(tariffs):
                    if i == 0 or txt != tariffs[i-1]:  # 첫 포인트이거나, 앞의 값과 다를 때만 표시
                        plt.annotate(f'{txt:.2f}', (iter_list[i], tariffs[i]), textcoords="offset points", xytext=(0,10), ha='center')

                plt.title(f'Tariff for "{industry}" from {exporter} to {importer} in Repeated Game')
                plt.xlabel('Iteration')
                plt.ylabel('Tariff')
                plt.grid(True)
                
                # Save the plot
                file_name = f"{output_dir}/tariff_{industry}_{exporter}_to_{importer}.png"
                plt.savefig(file_name)
                plt.close()

for exporter in var.countries:
    for importer in var.countries:
        if exporter != importer:
            welfares = welfare_history[exporter][importer][industry]

            plt.figure(figsize=(10, 6))
            plt.plot(iter_list, welfares, marker='o', color='#bb0a1e', linewidth = 2)
            #plt.ylim([1.0, 1.5])
            plt.grid(True, linestyle='--', alpha=0.7)


            for i, txt in enumerate(welfares):
                if i == 0 or txt != welfares[i-1]:  # 첫 포인트이거나, 앞의 값과 다를 때만 표시
                    plt.annotate(f'{txt:.1f}', (iter_list[i], welfares[i]), textcoords="offset points", xytext=(0,10), ha='center')

            plt.title(f'Welfare for {importer} in Repeated Game')
            plt.xlabel('Reapeated Game')
            plt.ylabel('Welfare')
            plt.grid(True)
            
            # Save the plot
            file_name = f"{output_dir}/welfare_{importer}.png" #같다는 것을 확인했으면 제목 수정하기
            plt.savefig(file_name)
            plt.close()


# # 임시 딕셔너리 생성
# temp_pi = var.pi.copy()
# temp_p = var.p_ijs.copy()
# temp_t = var.t.copy()
# temp_T = var.T.copy()
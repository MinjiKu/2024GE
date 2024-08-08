import sys
import os

# Get the absolute path of the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(script_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)
import var
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import welfare
import matplotlib.pyplot as plt
from scipy.optimize import root, minimize
from scipy.optimize import Bounds

from matplotlib import rcParams

# Times New Roman 폰트를 사용하도록 설정
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] + rcParams['font.serif']

var.fill_gamma()
var.fill_pi()
factual_pi = var.pi.copy() #factual var.pi 보존

var.fill_alpha()

# Initialize previous_tau globally
previous_tau = var.tau.copy()

def update_economic_variables(tau, j):
    # Update t based on the new tau values
    for i in var.countries:
            if i==j: continue
            for industry in var.industries:
                    # print("print t for", i, j, industry, ":", var.t[i][j][industry])
                    # print("print tau for", i, j, industry, ":",tau[i][j][industry])
                    var.t[i][j][industry] = max(var.tau[i][j][industry] - 100, 1e-10)

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

<<<<<<< HEAD
def gov_obj(tau, j):
    # Update economic variables based on the current tau_js
    update_economic_variables(tau, j)
=======
def gov_obj(tau_js, j):
>>>>>>> cabf1e5 (progress)

    total = 0
    for s in var.industries:
        total += var.pol_econ[j][s] * calc_welfare(j, s)
<<<<<<< HEAD

    #print("total (gov_obj) for",j, iter ,":",total)
=======
    #print("total: ")
    #print(total)
    return np.random.rand()
    return -total  # We minimize, so we return the negative
    total = 0
    tau_index = 0  # Initialize index for tau_js array

    for s in var.industries:
        # Print the current tau value and welfare for debugging
        print(f"Country: {j}, Industry: {s}, Tau Value: {tau_js[tau_index]}")

        welfare = calc_welfare(j, s)
        print(f"Welfare: {welfare}")

        tau_value = tau_js[tau_index]

        # Introduce non-linearity by squaring the welfare
        total += var.pol_econ[j][s] * (welfare ** 2)  # Square of welfare

        # Additional non-linearity: Exponential term based on tau_js
        tau_adjustment = np.exp(-abs(tau_value))  # Exponential decay term based on tau
        total += var.pol_econ[j][s] * tau_adjustment * welfare

        # Move to the next index for tau_js
        tau_index += 1

    # Print the final total for debugging
    print(f"Total Objective Value for Country {j}: {total}")

    total_penalty = np.sum(np.exp(-abs(tau_js)))  # Penalize high tariffs more
    total += total_penalty
>>>>>>> cabf1e5 (progress)

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
                          (var.P_hat[j][s] ** (var.sigma[s] - 1)) * (abs(var.tau_hat[i][j][s]) ** -var.sigma[s]) +
                          (var.pi[j][s] / x2(j) * var.pi_hat[j][s]))
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

def update_hats(tau, t, pi):
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

def flatten_tau(tau_dict):
    flat_list = []
    for importer, industries in tau_dict.items():
        for industry, value in industries.items():
            flat_list.extend(value.values())
    return flat_list


def unflatten_tau(flat_list, exporter_name):
    idx = 0
    unflattened_tau = {j: {s: 0 for s in var.industries} for j in var.countries if j != exporter_name}
    for importer in unflattened_tau:
        for industry in var.industries:
            unflattened_tau[importer][industry] = flat_list[idx]
            idx += 1
    return unflattened_tau


# Initialize a dictionary to store tariff values for each iteration
tariff_history = {exporter: {importer: {industry: [] for industry in var.industries} for importer in var.countries if importer != exporter} for exporter in var.countries}

# Initialize a dictionary to store welfare values for each iteration
welfare_history = {country: {industry: [] for industry in var.industries} for country in var.countries}


welfare_gains = {importer: {country: 0 for country in var.countries if importer != country} for importer in var.countries}

# Calculate the cooperative welfare objective
def calculate_welfare_gains():
    global welfare_gains

    var.coop_lambda()
    welfare_gains = {importer: {country: 0 for country in var.countries if importer != country} for importer in var.countries}

    for importer in var.countries:
       
        for country in var.countries:
            if importer != country:
                welfare_gains[importer][country] += sum(var.pol_econ[country][s] * calc_welfare(country, s) for s in var.industries)

        # print("welfare_gains")
        # print(welfare_gains)
        # print("\n")
    return welfare_gains

cooperative_welfare_target = 0
total_welfare_gain = 0

def calculate_cooperative_welfare_target():
    global cooperative_welfare_target, total_welfare_gain

    welfare_gains = calculate_welfare_gains()
    # Initialize total welfare gain
    total_welfare_gain = 0
    
<<<<<<< HEAD
    # Iterate over each importer and their associated welfare gains
    for importer in welfare_gains:
        for exporter in welfare_gains[importer]:
=======
#     # Create a DataFrame to organize the tariffs neatly
#     df_tau = pd.DataFrame({importer: {industry: var.tau[exporter][importer][industry] 
#                                     for industry in var.industries} 
#                         for importer in var.countries if importer != exporter})

#     # Print the DataFrame in the required format
#     print(df_tau.T.to_string())
# Run iterations until tariffs converge
for iteration in range(100):  # Arbitrary number of iterations
    print(f"\n--- Iteration {iteration + 1} ---")
    
    previous_tau = {exporter: {importer: {industry: var.tau[exporter][importer][industry] for industry in var.industries} for importer in var.countries if importer != exporter} for exporter in var.countries}

    for exporter in var.countries:
        tau, gov_obj_values = calculate_optimum_tariffs(exporter)

        print("gov_obj_values")
        print(gov_obj_values)

        for importer in tau:
            for industry in tau[importer]:
                tariff_history[exporter][importer][industry].append(tau[importer][industry])

        # Print the cooperative tariffs for the current iteration
        print(f"\nCooperative Tariffs after optimizing {exporter}:")
        df_tau = pd.DataFrame({importer: {industry: var.tau[exporter][importer][industry] 
                                          for industry in var.industries} 
                               for importer in var.countries if importer != exporter})
        print(df_tau.T.to_string())

    # Update welfare history
    for country in var.countries:
        for industry in var.industries:
            welfare_history[country][industry].append(calc_welfare(country, industry))

    # Check for convergence (e.g., using a small threshold for changes in tariffs)
    converged = True
    for exporter in var.countries:
        for importer in var.countries:
>>>>>>> cabf1e5 (progress)
            if importer == exporter:
                continue
            total_welfare_gain += welfare_gains[importer][exporter]
    
    print("Total welfare gain")
    print(total_welfare_gain)
    
    # Calculate the cooperative welfare target
    cooperative_welfare_target = total_welfare_gain / len(var.countries)
    print("Cooperative welfare target")
    print(cooperative_welfare_target)
    
    return cooperative_welfare_target

# Welfare function
def calc_welfare2(j, s, tau_dict):
    update_economic_variables(tau_dict, j)
    return calc_x(j, s) / var.P_j[j]


calculate_cooperative_welfare_target()

<<<<<<< HEAD
# Calculate the starting welfare for each country
starting_welfare = {country: 0 for country in var.countries}
for country in var.countries:
    for exporter in var.countries:
        if country == exporter: continue
        starting_welfare[country] += welfare_gains[country][exporter]

print("starting welfare")
print(starting_welfare)
print("\n")

# ========== logic checked =================

def cooperative_obj(tau_js, cooperative_welfare_target, j):
    # print(f"tau_js: {tau_js}")
    # print(f"cooperative_welfare_target: {cooperative_welfare_target}")
    # print(f"Country: {j}")

    tau_dict = unflatten_tau(tau_js, j)

    # print("tau_dict:", tau_dict)

    update_economic_variables(tau_dict, j)
    updated_welfare_gains = calculate_welfare_gains()
    total_welfare_for_j = sum(updated_welfare_gains[j].values())
    welfare_difference = total_welfare_for_j - cooperative_welfare_target
    return abs(welfare_difference)

# ----------------- initial ------------------

# # Flatten the tau dictionary into an array
# initial_tau = flatten_tau(var.tau_temp)

# bounds = Bounds(1,2)

# # Now run your minimization
# result = minimize(
#     cooperative_obj, 
#     initial_tau, 
#     args=(cooperative_welfare_target, 'Japan'),  # Replace 'Japan' with the actual country name if needed
#     method='L-BFGS-B', 
#     tol=1e-12,                 # Tolerance for termination (set very small to ensure high precision)
#     bounds = bounds,
#     options={'disp': True, 'maxiter': 20000, 'ftol': 1e-8}
# )

# print(result.x)

# -------------- changed ------------------
def optimize(country):
    initial_tau = flatten_tau(var.tau_temp)
    bounds = Bounds(1, 2)

    # Perform the minimization
    result = minimize(
        cooperative_obj,
        initial_tau,
        args=(cooperative_welfare_target, country),
        method='L-BFGS-B',
        tol=1e-12,
        bounds=bounds,
        options={'disp': True, 'maxiter': 20000, 'ftol': 1e-8}
    )
    return result

# Initialize a dictionary to store results for each country
optimization_results = {}

# Loop through all countries and perform optimization
for country in var.countries:
    print(f"Optimizing for {country}...")
    result = optimize(country)
    optimization_results[country] = result.x

    print(f"Optimization result for {country}:")
    print(optimization_results[country])

    print(f"welfare difference for {country}: ")
    print(result.fun)

#  ======================
# 1. Plot starting welfares for each country
plt.figure(figsize=(12, 6))
plt.bar(starting_welfare.keys(), starting_welfare.values(), color='skyblue')
plt.axhline(y=cooperative_welfare_target, color='red', linestyle='--', label='Cooperative Welfare Target')
plt.xlabel('Country')
plt.ylabel('Starting Welfare')
plt.title('Starting Welfares for Each Country')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('start&target.png')
# plt.show()

# # 2. Plot change rates of tau (after optimization)
# tau_change_rates = {}
# for country in var.countries:
#     initial_tau = flatten_tau(var.tau_temp)
#     optimized_tau_dict = unflatten_tau(optimization_results[country], country)
#     optimized_tau = flatten_tau(optimized_tau_dict)

#     change_rates = [(opt - init) / init if init != 0 else 0 for init, opt in zip(initial_tau, optimized_tau)]
#     tau_change_rates[country] = change_rates

# plt.figure(figsize=(12, 8))
# for country, change_rates in tau_change_rates.items():
#     plt.plot(change_rates, label=country)
# plt.xlabel('Tariff Index')
# plt.ylabel('Change Rate')
# plt.title('Change Rates of Tau After Optimization')
# plt.legend()
# plt.savefig('change_rate_plot.png')
# plt.tight_layout()
=======
# Plotting tariffs with logarithmic scale
def plot_tariff_history():
    for exporter in var.countries:
        for importer in var.countries:
            if importer == exporter:
                continue
            for industry in var.industries:
                plt.figure()
                plt.plot(tariff_history[exporter][importer][industry], label=f"{importer} - {industry}")
                plt.xlabel("Iteration")
                plt.ylabel("Tariff")
                plt.yscale('log')  # Set y-axis to logarithmic scale
                plt.legend()
                plt.title(f"Tariff Convergence for Exporter: {exporter}")
                plt.savefig(os.path.join(output_dir, f"tariff_convergence_{exporter}_{importer}_{industry}.png"))
                plt.close()

# Call the function to plot tariffs
plot_tariff_history()

def plot_welfare_history():
    for country in var.countries:
        plt.figure()
        for industry in var.industries:
            plt.plot(welfare_history[country][industry], label=f"{industry}")
        plt.xlabel("Iteration")
        plt.ylabel("Welfare")
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.legend()
        plt.title(f"Welfare Convergence for Country: {country}")
        plt.savefig(os.path.join(output_dir, f"welfare_convergence_{country}.png"))
        plt.close()

# Call the function to plot welfare
plot_welfare_history()
>>>>>>> cabf1e5 (progress)

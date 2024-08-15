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

TARGET_WELFARE = 18077910774.08308

# Times New Roman 폰트를 사용하도록 설정
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] + rcParams['font.serif']

var.fill_gamma()
var.fill_pi()
factual_pi = var.pi.copy() #factual var.pi 보존

var.fill_alpha()

# Initialize previous_tau globally
previous_tau = var.tau4.copy()

def update_economic_variables(tau, j):
    # Update t based on the new tau values

    for i in var.countries:
        if i==j: continue
        for industry in var.industries:
            if (tau[i][j][industry]<1 or tau[i][j][industry]>2): print("tau out of range")
            else: var.t[i][j][industry] = max(tau[i][j][industry] - 1.0, 1e-10)

    # Update p_ijs and T based on the new values of t
    for i in var.countries:
        for j in var.countries:
            if i != j:
                for industry in var.industries:
                    change_rate = (var.tau5[i][j][industry] - previous_tau[i][j][industry]) / previous_tau[i][j][industry] if previous_tau[i][j][industry] != 0 else 0
                    var.p_ijs[i][j][industry] *= (1 + change_rate)
                    var.T[i][j][industry] *= (1 - change_rate * var.de[industry])

    # Recalculate gamma, pi, and alpha based on the updated t values
    var.fill_gamma()
    var.fill_alpha()
    var.fill_pi()

def calc_x(i, j, s, tau):
    # update_economic_variables(tau, j)
    sum = 0
    TR = 0
    for i in var.countries:
        if i == j: continue
        TR += var.t[i][j][s] * var.T[i][j][s]
    # print("change rate")
    # print(var.tau[i][j][s] - previous_tau[i][j][s])
    # sum += (var.T[i][j][s] + var.de[s] * (var.tau3[i][j][s] - previous_tau[i][j][s]))
    
    sum = var.w[j] * var.L_js[j][s] + var.pi[j][s] + var.L_js[j][s]/var.L_j[j] * TR
    
    return sum

# Welfare function
def calc_welfare(i, j, s, tau):
    return calc_x(i, j, s, tau) / var.P_j[j]

# Initialize a dictionary to store tariff values for each iteration
tariff_history = {exporter: {importer: {industry: [] for industry in var.industries} for importer in var.countries if importer != exporter} for exporter in var.countries}

# Initialize a dictionary to store welfare values for each iteration
welfare_history = {country: {industry: [] for industry in var.industries} for country in var.countries}




# # Calculate the cooperative welfare objective
# def calculate_welfare_gains():
#     global welfare_gains_array

#     var.coop_lambda()
#     # Iterate over each exporter
#     for importer in var.countries:
#         row = []
#         for exporter in var.countries:
#             if exporter != importer:
#                 row.append(sum(var.pol_econ[importer][s] * calc_welfare(exporter, importer, s, var.tau) for s in var.industries))
#             else:
#                 row.append(None)  # or another placeholder if you don't want to include self-comparison
#         welfare_gains_array.append(row)

#     print("welfare_gains_array")
#     print(welfare_gains_array)

#     return welfare_gains_array

# Calculate the cooperative welfare objective
def calculate_welfare_gains():
    welfare_gains_array = []

    var.coop_lambda()
    # Iterate over each importer
    for importer in var.countries:
        total = 0
        for exporter in var.countries:
            if exporter != importer:
                total += sum(var.pol_econ[importer][s] * calc_welfare(exporter, importer, s, var.tau5) for s in var.industries)
        
        welfare_gains_array.append(total)

    # print("welfare_gains_array")
    # print(welfare_gains_array)

    return welfare_gains_array


cooperative_welfare_target = 0

def calc_total_welfare():
    welfare_gains_array = calculate_welfare_gains()
    total_welfare_gain = 0
    # Iterate over each importer and their associated welfare gains
    for i in range(5):
        total_welfare_gain += welfare_gains_array[i]
        
    return total_welfare_gain

total = calc_total_welfare()
print("staring total welfare")
print(total)

def calculate_cooperative_welfare_target():
    global cooperative_welfare_target

    total_w = calc_total_welfare()
    
    # Calculate the cooperative welfare target
    cooperative_welfare_target = total_w / len(var.countries)
    
    return cooperative_welfare_target

calculate_cooperative_welfare_target()
print("target")
print(cooperative_welfare_target)
print("\n")

# ========== logic checked =================

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


# total_welfare = {country: 0.0 for country in var.countries}
countries = ['China', 'Korea', 'Japan', 'USA', 'Germany']
total_welfare = [0, 0, 0, 0, 0]

def cooperative_obj(tau_js, target_welfare, j):
    # Update the economic variables based on the new tariffs
    update_economic_variables(tau_js, j)

    welfare_gains_array = calculate_welfare_gains()

    # print("welfare gains")
    # print(welfare_gains_array)

    idx = 0

    if j == 'China': idx = 0
    elif j == 'Korea': idx = 1
    elif j == 'Japan': idx = 2
    elif j == 'USA': idx = 3
    else: idx= 4
    
    # Compute the difference between calculated welfare and target welfare
    welfare_difference = welfare_gains_array[idx] - target_welfare

    # print("country")
    # print(j)
    # print(welfare_gains_array[idx])
    
    return welfare_difference


# -------------- changed ------------------

def optimize_for_importer(j):
    # Flatten the tau structure for the current importer
    initial_tau = flatten_dict({
        i: {s: var.tau5[i][j][s] for s in var.industries}
        for i in var.countries if i != j
    })
    # initial_tau = np.random.rand(3 * (4)) * 0.5 + 1.0

    print("Initial tariffs:", initial_tau)

    # Define the local objective function for the importer
    def importer_coop_obj(flat_tau, target_welfare, j):
        unflattened_tau = unflatten_dict(flat_tau, j)
        update_economic_variables(unflattened_tau, j)
        return cooperative_obj(unflattened_tau, target_welfare, j)

    bounds = Bounds(1, 2)

    # Perform the minimization
    result = minimize(
        importer_coop_obj,
        initial_tau,
        args=(TARGET_WELFARE, j),
        method='L-BFGS-B',
        tol=1,
        bounds=bounds,
        options={'disp': True, 'maxiter': 20000, 'ftol': 1e-8}
    )

    # Map the results back to the original tau structure
    optimized_tau = unflatten_dict(result.x, j)

    return optimized_tau

# Initialize a dictionary to store results for each country
optimization_results = {}
tariff_history = {i: {j: {industry: [] for industry in var.industries} for j in var.countries if j != i} for i in var.countries}
welfare_values = [0 for country in var.countries]

idx_j = 0
for j in var.countries:
    print(f"Optimizing for {j}...")
    optimized_tau = optimize_for_importer(j)
    print("Optimized tariffs:", optimized_tau)


    # Update global tariff values
    for i in var.countries:
        if i != j:
            for s in var.industries:
                var.tau5[i][j][s] = optimized_tau[i][j][s]
                tariff_history[i][j][s].append(max(optimized_tau[i][j][s], 1e-10))
            # Calculate and print the total welfare for the country `j`
            total_welfare = sum(calc_welfare(i, j, s, var.tau5) for s in var.industries)
            welfare_values[idx_j] += sum(calc_welfare(i, j, s, var.tau5) for s in var.industries)
    print(f"Total welfare for {j}: {total_welfare}")
    idx_j += 1

print("Final optimized tariffs:")
print(var.tau5)

# Assuming you have these values from the previous calculations
countries = ['China', 'Korea', 'Japan', 'USA', 'Germany']
# These would be the calculated welfare values for each country after optimization
# welfare_values = [sum(calc_welfare(i, country, s, var.tau3) for s in var.industries) for country in countries]
target_welfare = TARGET_WELFARE

# Plotting the graph
plt.figure(figsize=(10, 6))

# Plot the welfare values
plt.bar(countries, welfare_values, color='skyblue', label='Welfare')

# Plot the target welfare as a horizontal line
plt.axhline(y=target_welfare, color='r', linestyle='--', label='Target Welfare')

# Adding title and labels
plt.title('Welfare Comparison by Country')
plt.xlabel('Country')
plt.ylabel('Welfare')
plt.ylim(0, max(welfare_values + [target_welfare]) * 1.1)  # Adjust y-axis limit to accommodate all values

# Add legend
plt.legend()

# Show the plot
plt.show()

import sys
import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import Bounds

# Add parent directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import var

# Initialize variables
var.fill_gamma()
var.fill_pi()
factual_pi = var.pi.copy()  # Save factual pi
var.fill_alpha()

# Initialize global previous_tau
previous_tau = var.tau.copy()

# Update economic variables based on tau
def update_economic_variables(tau, j):
    for i in var.countries:
        if i == j:
            continue
        for industry in var.industries:
            var.t[i][j][industry] = max(tau[i][j][industry] - 100, 1e-10)

    var.fill_gamma()
    var.fill_alpha()
    var.fill_pi()

    for i in var.countries:
        for j in var.countries:
            if i != j:
                for industry in var.industries:
                    change_rate = (tau[i][j][industry] - previous_tau[i][j][industry]) / previous_tau[i][j][industry] if previous_tau[i][j][industry] != 0 else 0
                    var.p_ijs[i][j][industry] *= (1 + change_rate)
                    var.T[i][j][industry] *= (1 - change_rate * var.de[industry])

# Calculate welfare for a country and industry
def calc_x(j, s):
    TR = sum(var.t[i][j][s] * var.T[i][j][s] for i in var.countries if i != j)
    return var.w[j] * var.L_js[j][s] + var.pi[j][s] + var.L_js[j][s] / var.L_j[j] * TR

# Calculate welfare
def calc_welfare(j, s):
    return calc_x(j, s) / var.P_j[j]

# Government objective function
def gov_obj(tau, j):
    update_economic_variables(tau, j)
    return -sum(var.pol_econ[j][s] * calc_welfare(j, s) for s in var.industries)

# Constraint function eq_12
def eq_12(j, s):
    total = sum((var.gamma[i][j][s] * (var.tau_hat[i][j][s] ** (1 - var.sigma[s]))) ** (1 / (1 - var.sigma[s])) for i in var.countries if i != j)
    var.P_hat[j][s] = total
    return var.P_hat[j][s] - total

# Flatten tau dictionary into a list
def flatten_tau(tau_dict):
    return [value for industries in tau_dict.values() for value in industries.values()]

# Unflatten a list back into tau dictionary
def unflatten_tau(flat_list, exporter_name):
    idx = 0
    unflattened_tau = {j: {s: 0 for s in var.industries} for j in var.countries if j != exporter_name}
    for importer in unflattened_tau:
        for industry in var.industries:
            unflattened_tau[importer][industry] = flat_list[idx]
            idx += 1
    return unflattened_tau

# Calculate cooperative welfare gains
def calculate_welfare_gains():
    var.coop_lambda()
    welfare_gains = {importer: {country: 0 for country in var.countries if importer != country} for importer in var.countries}

    for importer in var.countries:
        for country in var.countries:
            if importer != country:
                welfare_gains[importer][country] += sum(var.pol_econ[country][s] * calc_welfare(country, s) for s in var.industries)
    return welfare_gains

# Calculate the cooperative welfare target
def calculate_cooperative_welfare_target():
    welfare_gains = calculate_welfare_gains()
    total_welfare_gain = sum(welfare_gains[importer][exporter] for importer in var.countries for exporter in var.countries if importer != exporter)
    return total_welfare_gain / len(var.countries)

cooperative_welfare_target = calculate_cooperative_welfare_target()

# Calculate starting welfare for each country
starting_welfare = {country: sum(var.pol_econ[country][s] * calc_welfare(country, s) for s in var.industries) for country in var.countries}

# Cooperative objective function
def cooperative_obj(tau_js, cooperative_welfare_target, j):
    tau_dict = unflatten_tau(tau_js, j)
    update_economic_variables(tau_dict, j)
    updated_welfare_gains = calculate_welfare_gains()
    total_welfare_for_j = sum(updated_welfare_gains[j].values())
    welfare_difference = total_welfare_for_j - cooperative_welfare_target
    return abs(welfare_difference)

# Optimization function for a single country
def optimize(country):
    initial_tau = flatten_tau(var.tau_temp)
    bounds = Bounds(1, 2)
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

# Dictionary to store results for each country
optimization_results = {}

# Optimize for each country
for country in var.countries:
    print(f"Optimizing for {country}...")
    result = optimize(country)
    optimization_results[country] = result.x
    print(f"Optimization result for {country}: {optimization_results[country]}")
    print(f"Welfare difference for {country}: {result.fun}")

# Plot starting welfare for each country
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

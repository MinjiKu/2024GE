import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

# Get the absolute path of the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(script_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)
import var
import welfare

# Set up Times New Roman font for plots
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] + rcParams['font.serif']

# Initialize previous_tau globally
previous_tau = var.tau.copy()

def update_economic_variables(tau, j):
    for i in var.countries:
        if i == j: continue
        for industry in var.industries:
            var.t[i][j][industry] = max(var.tau[i][j][industry] - 100, 1e-10)
    var.fill_gamma()
    var.fill_alpha()
    var.fill_pi()
    for i in var.countries:
        for j in var.countries:
            if i != j:
                for industry in var.industries:
                    change_rate = (var.tau[i][j][industry] - previous_tau[i][j][industry]) / previous_tau[i][j][industry] if previous_tau[i][j][industry] != 0 else 0
                    var.p_ijs[i][j][industry] *= (1 + change_rate)
                    var.T[i][j][industry] *= (1 - change_rate * var.de[industry])

def calc_x(j, s):
    TR = sum(var.t[i][j][s] * var.T[i][j][s] for i in var.countries if i != j)
    return var.w[j] * var.L_js[j][s] + var.pi[j][s] + var.L_js[j][s] / var.L_j[j] * TR

def calc_welfare(j, s):
    return calc_x(j, s) / var.P_j[j]

def calculate_welfare_gains():
    var.coop_lambda()
    welfare_gains = {importer: {country: 0 for country in var.countries if importer != country} for importer in var.countries}
    for importer in var.countries:
        for country in var.countries:
            if importer != country:
                welfare_gains[importer][country] += sum(var.pol_econ[country][s] * calc_welfare(country, s) for s in var.industries)
    return welfare_gains

def calculate_cooperative_welfare_target():
    global cooperative_welfare_target
    welfare_gains = calculate_welfare_gains()
    total_welfare_gain = sum(welfare_gains[importer][exporter] for importer in welfare_gains for exporter in welfare_gains[importer] if importer != exporter)
    cooperative_welfare_target = total_welfare_gain / len(var.countries)
    return cooperative_welfare_target

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
    return [value for importer in tau_dict for value in tau_dict[importer].values()]

def unflatten_tau(flat_list, exporter_name):
    idx = 0
    unflattened_tau = {j: {s: 0 for s in var.industries} for j in var.countries if j != exporter_name}
    for importer in unflattened_tau:
        for industry in var.industries:
            unflattened_tau[importer][industry] = flat_list[idx]
            idx += 1
    return unflattened_tau

def cooperative_obj(tau_js, cooperative_welfare_target, j):
    tau_dict = unflatten_tau(tau_js, j)
    update_economic_variables(tau_dict, j)
    updated_welfare_gains = calculate_welfare_gains()
    total_welfare_for_j = sum(updated_welfare_gains[j].values())
    welfare_difference = total_welfare_for_j - cooperative_welfare_target
    return abs(welfare_difference)

def optimize_for_country(country):
    # Flatten the tau dictionary for optimization
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

# Calculate the cooperative welfare target
calculate_cooperative_welfare_target()

# Initialize a dictionary to store results for each country
optimization_results = {}

# Loop through all countries and perform optimization
for country in var.countries:
    print(f"Optimizing for {country}...")
    result = optimize_for_country(country)
    optimization_results[country] = result
    print(f"Optimization result for {country}:")
    print(result)

# Save results to a file or process them further as needed

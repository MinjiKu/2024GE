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
from scipy.optimize import root

from matplotlib import rcParams

# Times New Roman 폰트를 사용하도록 설정
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] + rcParams['font.serif']

var.fill_gamma()
var.fill_pi()
factual_pi = var.pi.copy() #factual var.pi 보존

var.fill_alpha()
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

def gov_obj(tau, j):
    # Update economic variables based on the current tau_js
    update_economic_variables(tau, j)

    total = 0
    for s in var.industries:
        total += var.pol_econ[j][s] * calc_welfare(j, s)

    #print("total (gov_obj) for",j, iter ,":",total)

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

# def calculate_optimum_tariffs(exporter_name):
#     global tau, t, tariff_matrices

#     optimal_taus = {j: {industry: 0 for industry in var.industries} for j in var.countries if j != exporter_name}
#     gov_obj_values = {j: {industry: 0 for industry in var.industries} for j in var.countries if j != exporter_name} 
    
#     count_idx = 0
#     for j, importer in enumerate(var.countries):
#         if importer == exporter_name:
#             continue
        
#         # Calculate welfare gains and cooperative welfare target
#         welfare_gains = calculate_welfare_gains()
#         cooperative_welfare_target = calculate_cooperative_welfare_objective(welfare_gains)
        
#         flat_matrix = flatten_dict({j: {s: var.tau[exporter_name][j][s] for s in var.industries} for j in var.countries if j != exporter_name})
        
#         # result = minimize(
#         #     gov_obj, 
#         #     flat_matrix, 
#         #     args=(importer,), 
#         #     constraints=constraints_with_cooperative_welfare(flat_matrix, importer, cooperative_welfare_target)
#         # )
#         result = root(cooperative_obj, var.tau, cooperative_welfare_target)
        
#         idx = 0
#         for industry in var.industries:
#             optimal_taus[importer][industry] = result.x[count_idx * (var.num_industries) + idx]
#             gov_obj_values[importer][industry] = -result.fun
#             idx += 1
#         count_idx += 1

#     for importer in optimal_taus:
#         for industry in optimal_taus[importer]:
#             var.tau[exporter_name][importer][industry] = optimal_taus[importer][industry]
    
#     return optimal_taus, gov_obj_values

def flatten_tau(tau_dict):
    """
    Flatten a nested dictionary of tau into a single list.
    """
    flat_list = []
    for importer, industries in tau_dict.items():
        for industry, value in industries.items():
            flat_list.append(value)
    return flat_list

def unflatten_tau(flat_list, exporter_name):
    """
    Convert a flat list back into a nested dictionary of tau.
    """
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

# Calculate the cooperative welfare objective
def calculate_welfare_gains():
    var.coop_lambda()
    welfare_gains = {}
    for country in var.countries:
        welfare_gains[country] = sum(var.pol_econ[country][s] * calc_welfare(country, s) for s in var.industries)
    return welfare_gains

def calculate_cooperative_welfare_objective(welfare_gains):
    total_welfare_gain = sum(welfare_gains.values())
    cooperative_welfare_target = total_welfare_gain / len(var.countries)
    return cooperative_welfare_target

# def cooperative_obj(tau_js, cooperative_welfare_target, j):
#     total_welfare = sum(var.pol_econ[j][s] * calc_welfare(j, s) for s in var.industries)
#     return abs(total_welfare - cooperative_welfare_target)  # Absolute difference from the target

def cooperative_obj(tau_js, cooperative_welfare_target, exporter_name):
    """
    Calculate the residuals of the cooperative objective function.
    This function returns an array where each element represents
    the difference between the target and current welfare for each country.
    """
    residuals = []

    # Unflatten tau_js to the dictionary format
    tau_dict = unflatten_tau(tau_js, exporter_name)

    for j in var.countries:
        if j == exporter_name:
            continue
        
        # Calculate current welfare for the country
        total_welfare = sum(var.pol_econ[j][s] * calc_welfare(j, s) for s in var.industries)
        
        # Calculate the residual (difference from the cooperative welfare target)
        residual = total_welfare - cooperative_welfare_target
        residuals.append(residual)
    
    return np.array(residuals)


def update_economic_variables(tau_dict, exporter_name):
    """
    Update economic variables based on the new tau values.
    This function should modify the global variables like var.t, var.pi, etc.
    based on the new tau values.
    """
    global previous_tau
    previous_tau = {i: {j: {s: var.tau[i][j][s] for s in var.industries} for j in var.countries if j != i} for i in var.countries}

    for i in var.countries:
        if i == exporter_name:
            continue
        for s in var.industries:
            var.t[i][exporter_name][s] = max(tau_dict[i][s] - 100, 1e-10)

    var.fill_gamma()
    var.fill_alpha()
    var.fill_pi()

def cooperative_obj(tau_js, cooperative_welfare_target, exporter_name):
    """
    Calculate the residuals of the cooperative objective function.
    This function returns an array where each element represents
    the difference between the target and current welfare for each country.
    """
    residuals = []

    # Unflatten tau_js to the dictionary format
    tau_dict = unflatten_tau(tau_js, exporter_name)

    # Update economic variables based on the new tau values
    update_economic_variables(tau_dict, exporter_name)

    for j in var.countries:
        if j == exporter_name:
            continue
        
        # Calculate current welfare for the country
        total_welfare = sum(var.pol_econ[j][s] * calc_welfare(j, s) for s in var.industries)
        
        # Calculate the residual (difference from the cooperative welfare target)
        residual = total_welfare - cooperative_welfare_target
        for i in range(3):
            residuals.append(residual)
    
    return np.array(residuals)

def calculate_optimum_tariffs(exporter_name):
    global var

    # Calculate welfare gains and cooperative welfare target
    welfare_gains = calculate_welfare_gains()
    cooperative_welfare_target = calculate_cooperative_welfare_objective(welfare_gains)

    # Flatten initial guess for tau
    flat_matrix = flatten_tau({j: {s: var.tau[exporter_name][j][s] for s in var.industries} for j in var.countries if j != exporter_name})

    # Use root to solve for optimal tau
    result = root(lambda x: cooperative_obj(x, cooperative_welfare_target, exporter_name), flat_matrix)

    if result.success:
        optimal_tau = unflatten_tau(result.x, exporter_name)
        for importer in optimal_tau:
            for industry in optimal_tau[importer]:
                var.tau[exporter_name][importer][industry] = optimal_tau[importer][industry]
        print("Final cooperative tariffs:")
        print(var.tau[exporter_name])
    else:
        print("Optimization did not converge.")

    return var.tau[exporter_name]

exporter_name = 'USA'  # Replace with the relevant exporter country
final_tariffs = calculate_optimum_tariffs(exporter_name)


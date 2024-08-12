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
import pickle

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
            if (tau[i][j][industry]<1 or tau[i][j][industry]>2): print("tau out of range")
            else: var.t[i][j][industry] = max(tau[i][j][industry] - 1.0, 1e-10)

    # Update p_ijs and T based on the new values of t
    for i in var.countries:
        for j in var.countries:
            if i != j:
                for industry in var.industries:
                    change_rate = (var.tau[i][j][industry] - previous_tau[i][j][industry]) / previous_tau[i][j][industry] if previous_tau[i][j][industry] != 0 else 0
                    var.p_ijs[i][j][industry] *= (1 + change_rate)
                    var.T[i][j][industry] *= (1 - change_rate * var.de[industry])

    # Recalculate gamma, pi, and alpha based on the updated t values
    var.fill_gamma()
    var.fill_alpha()
    var.fill_pi()

def calc_x(i, j, s, tau):
    sum = 0
    TR = 0
    # for i in var.countries:
    #     if i == j: continue
    TR += var.t[i][j][s] * var.T[i][j][s]
    # sum += (var.T[i][j][s] + var.de[s] * (var.tau[i][j][s] - previous_tau[i][j][s]))
    
    sum = var.w[j] * var.L_js[j][s] + var.pi[j][s] + var.L_js[j][s]/var.L_j[j] * TR
    
    return sum

# Welfare function
def calc_welfare(i, j, s, tau):
    return calc_x(i, j, s, tau) / var.P_j[j]

# Initialize a dictionary to store tariff values for each iteration
tariff_history = {exporter: {importer: {industry: [] for industry in var.industries} for importer in var.countries if importer != exporter} for exporter in var.countries}

# Initialize a dictionary to store welfare values for each iteration
welfare_history = {country: {industry: [] for industry in var.industries} for country in var.countries}


welfare_gains = {exporter: {country: 0 for country in var.countries if exporter != country} for exporter in var.countries}

# Calculate the cooperative welfare objective
def calculate_welfare_gains():
    global welfare_gains

    var.coop_lambda()
    welfare_gains = {exporter: {country: 0 for country in var.countries if exporter != country} for exporter in var.countries}

    for importer in var.countries:
        for country in var.countries:
            if importer != country:
                welfare_gains[country][importer] += sum(var.pol_econ[importer][s] * calc_welfare(country, importer, s, var.tau) for s in var.industries)

    return welfare_gains

cooperative_welfare_target = 0
total_welfare_gain = 0

def calc_total_welfare():
    global total_welfare_gain
    # Iterate over each importer and their associated welfare gains
    for importer in welfare_gains:
        for exporter in welfare_gains[importer]:
            if importer == exporter:
                continue
            total_welfare_gain += welfare_gains[exporter][importer]

def calculate_cooperative_welfare_target():
    global cooperative_welfare_target

    welfare_gains = calculate_welfare_gains()

    calc_total_welfare()
    
    # Calculate the cooperative welfare target
    cooperative_welfare_target = total_welfare_gain / len(var.countries)
    
    return cooperative_welfare_target

calculate_cooperative_welfare_target()

changing_welfare = {}

def calculate_welfare_change():
    global changing_welfare
    # Calculate the starting welfare for each country
    changing_welfare = {country: 0 for country in var.countries}
    
    calculate_welfare_gains()

    print("welfare_gains")
    print(welfare_gains)
    for country in var.countries:
        for exporter in var.countries:
            if country == exporter: continue
            changing_welfare[country] += welfare_gains[exporter][country]

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


def cooperative_obj(tau_js, cooperative_welfare_target, j):
    calculate_welfare_change()
    welfare_difference = changing_welfare[j] - cooperative_welfare_target
    return abs(welfare_difference)


# -------------- changed ------------------
def optimize_for_importer(j):
    # Flatten the tau structure for the current importer
    initial_tau = flatten_dict({
        i: {s: var.tau[i][j][s] for s in var.industries}
        for i in var.countries if i != j
    })

    # Define the local objective function for the importer
    def importer_coop_obj(flat_tau, cooperative_welfare_target, j):
        unflattened_tau = unflatten_dict(flat_tau, j)
        update_economic_variables(unflattened_tau, j)
        return cooperative_obj(unflattened_tau, cooperative_welfare_target, j)

    bounds = Bounds(1, 2)

    # Perform the minimization
    result = minimize(
        importer_coop_obj,
        initial_tau,
        args=(cooperative_welfare_target, j),
        method='L-BFGS-B',
        tol=1e-12,
        bounds=bounds,
        options={'disp': True, 'maxiter': 20000, 'ftol': 1e-8}
    )

    # return result.x
    # Map the results back to the original tau structure
    # optimized_tau = unflatten_tau(result.x, j)
    optimized_tau = unflatten_dict(result.x, j)

    return optimized_tau

# Initialize a dictionary to store results for each country
optimization_results = {}
tariff_history = {i: {j: {industry: [] for industry in var.industries} for j in var.countries if j != i} for i in var.countries}

iteration = 15

# Perform 100 iterations
for iter in range(iteration):
    print(f"Iteration {iter + 1}")
    # Loop through all countries and perform optimization
    for j in var.countries:
        print(f"Optimizing for {j}...")
        optimized_tau = optimize_for_importer(j)

        print(f"Optimization result for {j}:")
        print(optimized_tau)

        for i in var.countries:
            if i != j:
                for s in var.industries:
                    var.tau[i][j][s] = optimized_tau[i][j][s]
                    tariff_history[i][j][s].append(max(optimized_tau[i][j][s], 1e-10)) 


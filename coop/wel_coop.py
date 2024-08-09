import sys
import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize

# Set font for matplotlib
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] + rcParams['font.serif']

# Get the absolute path of the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import var

# Initialize economic variables
var.fill_gamma()
var.fill_pi()
factual_pi = var.pi.copy()

def flatten_dict(tau_dict):
    flat_list = []
    for i in tau_dict:
        for j in tau_dict:
            if i != j:
                for industry in var.industries:
                    flat_list.append(tau_dict[i][j][industry])
    return torch.tensor(flat_list, dtype=torch.float32)

def unflatten_dict(flat_list, j):
    unflattened_dict = {}
    index = 0
    for i in var.countries:
        if i != j:
            unflattened_dict[i] = {j: {}}
            for industry in var.industries:
                unflattened_dict[i][j][industry] = flat_list[index]
                index += 1
    return unflattened_dict

def update_economic_variables(tau, j):
    for i in tau.keys():
        if i == j: continue
        for industry in var.industries:
            var.t[i][j][industry] = max(tau[i][j][industry] - 1, 1e-10)
    var.fill_gamma()
    var.fill_alpha()
    var.fill_pi()
    for i in var.countries:
        for j in var.countries:
            if i != j:
                for industry in var.industries:
                    previous_tau = var.factual_tau[i][j][industry]
                    change_rate = (var.tau[i][j][industry] - previous_tau) / previous_tau if previous_tau != 0 else 0
                    var.p_ijs[i][j][industry] *= (1 + change_rate)
                    var.T[i][j][industry] *= (1 + change_rate * var.de[industry])

def calc_x(j, s):
    TR = sum(var.t[i][j][s] * var.T[i][j][s] for i in var.countries if i != j)
    return var.w[j] * var.L_js[j][s] + var.pi[j][s] + var.L_js[j][s] / var.L_j[j] * TR

def calc_welfare(j, s):
    return calc_x(j, s) / var.P_j[j]

def gov_obj(tau, j):
    update_economic_variables(tau, j)
    total = sum(var.pol_econ[j][s] * calc_welfare(j, s) for s in var.industries)
    return -total

def optimize_for_importer(j, var):
    initial_tau_dict = {i: {j: {s: var.tau[i][j][s] for s in var.industries}} for i in var.countries if i != j}
    initial_tau = flatten_dict(initial_tau_dict)
    
    def importer_gov_obj(flat_tau):
        unflattened_tau = unflatten_dict(flat_tau, j)
        update_economic_variables(unflattened_tau, j)
        return gov_obj(unflattened_tau, j)
    
    result = minimize(importer_gov_obj, initial_tau)
    optimized_tau = unflatten_dict(result.x, j)
    
    for i in var.countries:
        if i != j:
            for s in var.industries:
                var.tau[i][j][s] = optimized_tau[i][j][s]

def independent_optimization(var):
    for iter in range(20):
        print(f"Iteration {iter + 1}")
        for j in var.countries:
            print(f"Optimizing tariffs for imports into country: {j}")
            optimize_for_importer(j, var)
        
        temp_p = var.p_ijs.copy()
        temp_T = var.T.copy()
        temp_pi = var.pi.copy()

        update_economic_variables(var.tau, j)

        p_is = calculate_p_is(var.p_ijs)
        p_js = calculate_p_js(var.p_ijs)

        var.fill_pi()
        var.fill_gamma()
        var.fill_alpha()

        update_hats(var.tau, var.t, var.pi)

        changerate_pi = {country: {industry: (var.pi[country][industry] - temp_pi[country][industry]) / temp_pi[country][industry] for industry in var.industries} for country in var.countries}
        changerate_p_is = {i: {industry: (p_is[i][industry] - temp_p_is[i][industry]) / temp_p_is[i][industry] for industry in var.industries} for i in var.countries}
        changerate_p_js = {j: {industry: (p_js[j][industry] - temp_p_js[j][industry]) / temp_p_js[j][industry] for industry in var.industries} for j in var.countries}
        changerate_T = {i: {j: {industry: (var.T[i][j][industry] - temp_T[i][j][industry]) / temp_T[i][j][industry] for industry in var.industries} for j in var.countries if i != j} for i in var.countries}

        print(f"Welfare change effect of iteration {iter + 1}: {welfare_change()}")

def calculate_p_js(p_ijs):
    p_js = {i: {s: max(p_ijs[i][j][s] for j in var.countries if j != i) for s in var.industries} for i in var.countries}
    return p_js

def calculate_p_is(p_ijs):
    p_is = {j: {s: min(p_ijs[i][j][s] if p_ijs[i][j][s] != 0 else 1e-10 for i in var.countries if i != j) for s in var.industries} for j in var.countries}
    return p_is

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

def welfare_change():
    delta_W_W = {}
    x_js_list = {country: {industry: calc_x(country, industry) for industry in var.industries} for country in var.countries}
    x_j_list = {country: sum(x_js_list[country][s] for s in var.industries) for country in var.countries}
    
    for j in var.countries:
        term1 = sum((var.T[i][j][s] / x_j_list[j]) * (changerate_p_js[j][s] - changerate_p_js[i][s]) for i in var.countries if i != j for s in var.industries)
        term2 = sum((var.pi[j][s] / x_j_list[j]) * (changerate_pi[j][s] - changerate_p_js[j][s]) for s in var.industries)
        term3 = sum((var.t[i][j][s] * var.T[i][j][s] / x_j_list[j]) * (changerate_T[i][j][s] - changerate_p_is[i][s]) for i in var.countries if i != j for s in var.industries)
        delta_W_W[j] = term1 + term2 + term3
    
    return delta_W_W

# Perform independent optimization for each country
independent_optimization(var)

# Print results
for i in var.countries:
    print(f"\nTariffs for {i} as country_i:")
    df_tau = pd.DataFrame({j: {s: var.tau[i][j][s] for s in var.industries} for j in var.countries if j != i})
    print(df_tau)

    print(f"\nt values for {i} as the home country:")
    df_t = pd.DataFrame({j: {s: var.t[i][j][s] for s in var.industries} for j in var.countries if j != i})
    print(df_t)

# Plot tariff history
iter_list = list(range(1, 21))
for exporter in var.countries:
    for importer in var.countries:
        if exporter != importer:
            for industry in var.industries:
                tariffs = tariff_history[exporter][importer][industry]

                plt.figure(figsize=(10, 6))
                plt.plot(iter_list, tariffs, marker='o', color='#bb0a1e', linewidth=2)
                plt.ylim([0, np.max(tariffs) + 0.5])
                plt.title(f'Tariff History for {exporter} to {importer} in {industry}')
                plt.xlabel('Iteration')
                plt.ylabel('Tariff')
                plt.grid(True)
                plt.savefig(f'tariff_history_{exporter}_{importer}_{industry}.png')
                plt.close()

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
    print("this is update econ vars")
    print("update tau value")
    print(tau)
    # unflatten_dict(tau, j)
    for i in var.countries:
            if i==j: continue
            for industry in var.industries:
                    print("print t for", i, j, industry, ":", var.t[i][j][industry])
                    print("print tau for", i, j, industry, ":", tau[i][j][industry])
                    # var.t[i][j][industry] = max(var.tau[i][j][industry] - 100, 1e-10)

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
        # if j == 'China' or j == 'USA':
        sum += (var.T[i][j][s] + var.de[s] * (var.tau[i][j][s] - previous_tau[i][j][s]))
        # else:
        #     sum += (var.T[i][j][s] + var.de[s] * (var.tau2[i][j][s] - var.tau[i][j][s]))
    
    # sum = var.w[j] * var.L_js[j][s] + var.pi[j][s] + var.L_js[j][s]/var.L_j[j] * TR
    
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

# def flatten_tau(tau_dict):
#     flat_list = []
#     for importer, industries in tau_dict.items():
#         for industry, value in industries.items():
#             flat_list.extend(value.values())
#     return flat_list


# def unflatten_tau(flat_list, importer_name):
#     # idx = 0
#     # unflattened_tau = {j: {s: 0 for s in var.industries} for j in var.countries if j != importer_name}
#     # for importer in unflattened_tau:
#     #     for industry in var.industries:
#     #         unflattened_tau[importer][industry] = flat_list[idx]
#     #         idx += 1
#     # return unflattened_tau
#     unflattened_dict = {}
#     index = 0
    
#     for i in var.countries:
#         for j in var.countries:  # Iterate over all exporter keys
#             # if i == importer_name:  continue # Skip the importer itself
#             if i == j: continue
#             unflattened_dict[i] = {j: {}}  # Initialize the structure
#             for industry in var.industries:  # Iterate over all industries
#                 unflattened_dict[i][j][industry] = flat_list[index]
#                 index += 1

#     return unflattened_dict


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
    
    # Iterate over each importer and their associated welfare gains
    for importer in welfare_gains:
        for exporter in welfare_gains[importer]:
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

starting_welfare = {}

def calculate_welfare_change():
    global starting_welfare
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

def flatten_dict(tau_dict, country):
    flat_list = []
    for i in tau_dict.keys():  # Iterate over all exporter keys
        if i == country: continue
        for industry in var.industries:  # Iterate over all industries
            flat_list.append(tau_dict[country][i][industry])
    return flat_list

def unflatten_dict(flat_list, j):
    print(f"Called unflatten_dict with flat_list of type {type(flat_list)}")
    unflattened_dict = {}
    index = 0

    print("flat list")
    print(flat_list)
    print("\n")
    
    for i in var.countries:  # Iterate over all exporter keys
        if i != j:  # Skip the importer itself
            unflattened_dict[i] = {j: {}}  # Initialize the structure
            for industry in var.industries:  # Iterate over all industries
                unflattened_dict[i][j][industry] = flat_list[index] 
                index += 1
    
    return unflattened_dict

# helper functions

def cooperative_obj(tau_js, cooperative_welfare_target, j):
    print(f"tau_js: {tau_js}")
    print(f"cooperative_welfare_target: {cooperative_welfare_target}")
    print(f"Country: {j}")

    tau_dict = unflatten_dict(tau_js, j)

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
    print("tau")
    print(var.tau)
    print("\n")
    initial_tau = flatten_dict(var.tau, country)
    print("initial tau")
    print(initial_tau)
    print("\n")

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
    print("result")
    print(result.x)
    print("\n")
    # return result.x
    # Map the results back to the original tau structure
    # optimized_tau = unflatten_tau(result.x, country)
    optimized_tau = unflatten_dict(result.x, country)

    print("optimized tau")
    print(optimized_tau)
    print("\n")

    return optimized_tau

# Initialize a dictionary to store results for each country
optimization_results = {}

# Loop through all countries and perform optimization
for country in var.countries:
    print(f"Optimizing for {country}...")
    result = optimize(country)
    optimization_results[country] = result
    update_economic_variables(result, country)

    print(f"Optimization result for {country}:")
    print(optimization_results[country])

    print("optimization result")
    print(optimization_results)

    # print(f"welfare difference for {country}: ")
    # print(result.fun)

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

# ========================


# iteration = 10

# # Perform 100 iterations
# for iter in range(iteration):
#     print(f"Iteration {iter + 1}")
#     print(var.tau)

#     previous_tau = var.tau.copy()
#     # Store the current state of the economic variables
#     # temp_p = var.p_ijs.copy() 
#     temp_T = var.T.copy()
#     temp_pi = var.pi.copy()
#     # temp_p_is = calculate_p_is(var.p_ijs)
#     # temp_p_js = var.p_ijs.copy() 
#     # p_is = calculate_p_is(var.p_ijs)
#     # p_js = var.p_ijs.copy()
#     # Iterate over each importing country and optimize
    
    
#     for j in var.countries:
#         print(f"Optimizing tariffs for imports into country: {j}")
#         optimized_tau = optimize(j)

#         unflatten_dict(optimized_tau, j)

#         print("optimized_tau")
#         print(optimized_tau)
#         print("\n")

#         # #print(f"Welfare for {j} after optimization: {optimized_g}")
#         # welfare_history[j].append(optimized_g)
#         #print(f"print welfare history for {j}: {welfare_history[j]}")
#         for i in var.countries:
#             if i == j: continue
#             for s in var.industries:   
#                 # var.tau[i][j][s] = optimized_tau[i][s]
#                 # tariff_history[i][j][s].append(max(optimized_tau[i][s], 1e-10)) 
#                 var.tau[i][j][s] = optimized_tau[i][j][s]
#                 tariff_history[i][j][s].append(max(optimized_tau[i][j][s], 1e-10)) 
    

#     # Update economic variables with the new tau after all optimizations
#     update_economic_variables(var.tau, j)

#     # Recalculate p_is and p_js after updating tau
    
#     #Delta 값 계산
#     changerate_pi = {country: {industry: (var.pi[country][industry] - temp_pi[country][industry])/temp_pi[country][industry] for industry in var.industries} for country in var.countries}
#     # changerate_p_is = {i: {industry: (p_is[i][industry] - temp_p_is[i][industry])/ temp_p_is[i][industry] for industry in var.industries} for i in var.countries}
#     # changerate_p_js = {j: {i:{industry: (p_js[j][i][industry] - temp_p_js[j][i][industry])/temp_p_js[j][i][industry] for industry in var.industries} for i in var.countries if i!=j} for j in var.countries}
#     changerate_T = {i: {j: {industry: (var.T[i][j][industry] - temp_T[i][j][industry])/temp_T[i][j][industry] for industry in var.industries} for j in var.countries if i != j} for i in var.countries}
    
#     print("pi difference:" , ((var.pi[i][industry] - temp_pi[i][industry]) for i in var.countries for industry in var.industries))
#     # print("p_is difference:" , ((p_is[i][industry] - temp_p_is[i][industry]) for i in var.countries for industry in var.industries))
#     print("T difference:" , ((var.T[i][j][industry] - temp_T[i][j][industry]) for i in var.countries for j in var.countries if i!=j for industry in var.industries))
#     # print("p_js difference:" , ((p_js[i][j][industry] - p_js[i][j][industry]) for i in var.countries for j in var.countries if i!=j for industry in var.industries))
    
#     var.fill_pi()
#     var.fill_gamma()
#     var.fill_alpha()

#     # Recalculate gamma, var.pi, and alpha with new tau values
#     update_hats(var.tau, var.t, var.pi)


#     # # Call welfare_change with updated delta values
#     # print("welfare change effect of iteration", (iter+1), welfare_change())
#     print("\nCorresponding t values:")
#     for i in var.countries:
#         print(f"\nt values for {i} as the home country:")
#         df_t = pd.DataFrame({j: {s: var.t[i][j][s] for s in var.industries} for j in var.countries if j != i})
#         print(df_t)

#     # Print the current state of var.tau
#     print("Cooperative Tariffs (tau) after iteration", iter + 1)
#     for i in var.countries:
#         print(f"\nTariffs for {i} as the home country:")
#         df_tau = pd.DataFrame({j: {s: var.tau[i][j][s] for s in var.industries} for j in var.countries if j != i})
#         print(df_tau)

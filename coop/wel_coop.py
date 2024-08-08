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
import welfare
import matplotlib.pyplot as plt

var.fill_gamma()
var.fill_pi()
factual_pi = var.pi.copy() # factual var.pi 보존

var.fill_alpha()

def calc_x(j, s):
    sum = 0
    TR = 0
    for i in var.countries:
        if i == j:
            continue
        TR += var.t[i][j][s] * var.T[i][j][s]

    sum = var.w[j] * var.L_js[j][s] + var.pi[j][s] + var.L_js[j][s] / var.L_j[j] * TR

    return sum

# Welfare function
def calc_welfare(j, s):
    return calc_x(j, s) / var.P_j[j]

def gov_obj(tau_js, j):
    total = 0
    for s in var.industries:
        total += var.pol_econ[j][s] * calc_welfare(j, s)
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

def constraints(tau_js, j):
    cons = []

    # Constraint 1: eq_12
    for s in var.industries:
        cons.append({'type': 'eq', 'fun': lambda tau_js, j=j, s=s: eq_12(j, s)})

    # Constraint 2: eq_13
    cons.append({'type': 'eq', 'fun': lambda tau_js, j=j: eq_13(j)})

    # Constraint 3: eq_10 for each country i and industry s
    for i in var.countries:
        if i != j:
            for s in var.industries:
                cons.append({'type': 'eq', 'fun': lambda tau_js, i=i, s=s: eq_10(i, s)})

    return cons

def flatten_dict(dict_matrix):
    """
    Flatten the nested dictionary structure of tau into a single list of values.
    """
    flat_list = []
    for importer, industries in dict_matrix.items():
        for industry, value in industries.items():
            flat_list.append(value)
    return flat_list


def cal_x_j(country):
    sum = 0
    for industry in var.industries:
        sum += var.x[country][industry]
    return sum

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

def calculate_optimum_tariffs(exporter_name):
    global tau, t, tariff_matrices

    optimal_taus = {j: {industry: 0 for industry in var.industries} for j in var.countries if j != exporter_name}
    gov_obj_values = {}  # To store gov_obj values for each importer

    idx = 0
    for j, importer in enumerate(var.countries):
        if importer == exporter_name:
            continue

        # flat_matrix needs to get exporter_idx data
        flat_matrix = flatten_dict({j: {s: var.tau[exporter_name][j][s] for s in var.industries} for j in var.countries if j != exporter_name})

        result = minimize(gov_obj, flat_matrix, args=(importer,), constraints=constraints(flat_matrix, importer))

        line_idx = 0
        for industry in var.industries:
            optimal_taus[importer][industry] = result.x[line_idx * (var.num_countries - 1) + idx]
            line_idx += 1

        gov_obj_values[importer] = -result.fun  # Store the minimized value of gov_obj
        idx += 1

    var.tau[exporter_name] = optimal_taus
    var.fill_gamma()

    return optimal_taus, gov_obj_values

# Initialize a dictionary to store tariff values for each iteration
tariff_history = {i: {j: {industry: [] for industry in var.industries} for j in var.countries if j != i} for i in var.countries}

# Ensure the directory exists
output_dir = "nash_img"
os.makedirs(output_dir, exist_ok=True)

# Initialize a dictionary to store welfare values for each iteration
welfare_history = {country: {industry: [] for industry in var.industries} for country in var.countries}

# # Perform iterations
# iteration = 25
# for iteration in range(iteration):
#     print(f"Iteration {iteration + 1}")

#     new_taus = {i: {j: {industry: 0 for industry in var.industries} for j in var.countries if j != i} for i in var.countries}

#     flat_matrices = [flatten(tariff_matrices[i]) for i in range(len(var.countries))]

#     for k, country in enumerate(var.countries):
#         new_taus[country] = calculate_optimum_tariffs(country)

#     for i in var.countries:
#         for j in var.countries:
#             if j != i:
#                 for industry in var.industries:
#                     var.tau[i][j][industry] = new_taus[i][j][industry]
#                     tariff_history[i][j][industry].append(var.tau[i][j][industry])
#                     new_t = var.tau[i][j][industry] - 1
#                     var.t[i][j][industry] = max(new_t, 1e-10)

#     # Store the updated tau values as the new factual_tau for the next iteration
#     factual_tau = var.tau.copy()

#     # Calculate and update var.pi based on the new tariffs
#     var.fill_pi()

#     # Update hats after recalculating var.pi
#     update_hats(var.tau, var.t, var.pi)

#     # Calculate welfare for each country and industry
#     for country in var.countries:
#         for industry in var.industries:
#             welfare_value = calc_welfare(country, industry)
#             welfare_history[country][industry].append(welfare_value)

#     # Plot and save the graphs for each industry
#     for industry in var.industries:
#         for country in var.countries:
#             plt.plot(range(1, iteration + 2), [welfare_history[country][industry][i] for i in range(iteration + 1)], label=country)
#         plt.xlabel('Iteration')
#         plt.ylabel(f'Welfare in {industry}')
#         plt.title(f'Welfare Changes in {industry} Over Iterations')
#         plt.legend()
#         plt.savefig(os.path.join(output_dir, f'{industry}_welfare_iteration_{iteration + 1}.png'))
#         plt.clf()  # Clear the plot for the next industry

# print("Tariff history:")
# for i in var.countries:
#     for j in var.countries:
#         if j != i:
#             for industry in var.industries:
#                 print(f"Country {i} to Country {j}, Industry {industry}: {tariff_history[i][j][industry]}")

# Calculate the cooperative welfare objective
def calculate_welfare_gains():
    welfare_gains = {}
    for country in var.countries:
        welfare_gains[country] = sum(var.pol_econ[country][s] * calc_welfare(country, s) for s in var.industries)
    return welfare_gains

def calculate_cooperative_welfare_objective(welfare_gains):
    total_welfare_gain = sum(welfare_gains.values())
    cooperative_welfare_target = total_welfare_gain / len(var.countries)
    return cooperative_welfare_target

def cooperative_obj(tau_js, cooperative_welfare_target, j):
    total_welfare = sum(var.pol_econ[j][s] * calc_welfare(j, s) for s in var.industries)
    return abs(total_welfare - cooperative_welfare_target)

def calculate_cooperative_tariffs(cooperative_welfare_target):
    cooperative_taus = {i: {j: {industry: 0 for industry in var.industries} for j in var.countries if j != i} for i in var.countries}
    
    for k, country in enumerate(var.countries):
        flat_matrix = flat_matrices[k]
        
        result = minimize(cooperative_obj, flat_matrix, args=(cooperative_welfare_target, country), constraints=constraints(flat_matrix, country))
        
        idx = 0
        for j in var.countries:
            if j != country:
                for industry in var.industries:
                    cooperative_taus[country][j][industry] = result.x[idx]
                    idx += 1

    return cooperative_taus

# # Integrating cooperative tariffs calculation
# iteration = 10
# for iter in range(iteration):
#     print(f"Iteration {iter + 1}")
def cooperative_tariff():
    new_taus = {i: {j: {industry: 0 for industry in var.industries} for j in var.countries if j != i} for i in var.countries}
    
    for k, country in enumerate(var.countries):
        new_taus[country] = calculate_optimum_tariffs(country)
    
    # Store Nash tariffs
    for i in var.countries:
        for j in var.countries:
            if j != i:
                for industry in var.industries:
                    var.tau[i][j][industry] = new_taus[i][j][industry]
                    tariff_history[i][j][industry].append(var.tau[i][j][industry])
                    new_t = var.tau[i][j][industry] - 1
                    var.t[i][j][industry] = max(new_t, 1e-10)
    
    # Recalculate gamma, var.pi, and alpha with new tau values
    update_hats(var.tau, var.t, var.pi)
    
    # Calculate welfare gains and cooperative welfare target
    welfare_gains = calculate_welfare_gains()
    cooperative_welfare_target = calculate_cooperative_welfare_objective(welfare_gains)
    
    # Calculate cooperative tariffs
    cooperative_taus = calculate_cooperative_tariffs(cooperative_welfare_target)
    
    # Apply cooperative tariffs
    for i in var.countries:
        for j in var.countries:
            if j != i:
                for industry in var.industries:
                    var.tau[i][j][industry] = cooperative_taus[i][j][industry]
                    new_t = var.tau[i][j][industry] - 1
                    var.t[i][j][industry] = max(new_t, 1e-10)
    
    # # Recalculate gamma, var.pi, and alpha with cooperative tau values
    # update_hats(var.tau, var.t, var.pi)
    
    # # Calculate and print welfare change
    # delta_pi = {country: {industry: var.pi[country][industry] - temp_pi[country][industry] for industry in var.industries} for country in var.countries}
    # delta_p = {i: {j: {industry: var.p_is[i][j][industry] - temp_p[i][j][industry] for industry in var.industries} for j in var.countries if i != j} for i in var.countries}
    # delta_T = {i: {j: {industry: var.T[i][j][industry] - temp_T[i][j][industry] for industry in var.industries} for j in var.countries if i != j} for i in var.countries}

    # welfare_change(var.T, var.x, delta_p, var.p_is, var.pi, var.t, delta_pi, delta_T)
    # print("Welfare change: ", welfare_change(var.T, var.x, delta_p, var.p_is, var.pi, var.t, delta_pi, delta_T))
    # print("\n")

    # temp_pi = var.pi.copy()
    # temp_p = var.p_is.copy()
    # temp_t = var.t.copy()
    # temp_T = var.T.copy()

cooperative_tariff()

# Print the final cooperative tariffs
print("Cooperative Tariffs (tau):")
for i in var.countries:
    print(f"\nTariffs for {i} as the home country:")
    df_tau = pd.DataFrame({j: {s: var.tau[i][j][s] for s in var.industries} for j in var.countries if j != i})
    print(df_tau)

# cooperative tariff does not change over the iteration.

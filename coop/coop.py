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

var.fill_gamma()
var.fill_pi()
factual_pi = var.pi.copy()  # factual var.pi 보존

var.fill_alpha()

def calc_x(j, s):
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

    # 각 국가에 대해 고유의 관세율 매트릭스를 사용하도록 수정
    optimal_taus = {j: {industry: 0 for industry in var.industries} for j in var.countries if j != exporter_name}
    gov_obj_values = {j: {industry: 0 for industry in var.industries} for j in var.countries if j != exporter_name} 
    
    #exporter_idx = var.countries.index(exporter_name)
    count_idx = 0
    for j, importer in enumerate(var.countries):
        if importer == exporter_name:
            continue
        # flat_matrix는 실제로는 exporter_idx에 해당하는 데이터를 가져와야 합니다.
        flat_matrix = flatten_dict({j: {s: var.tau[exporter_name][j][s] for s in var.industries} for j in var.countries if j != exporter_name})
        
        result = minimize(gov_obj, flat_matrix, args=(importer,), constraints=constraints(flat_matrix, importer))
        
        idx = 0
        for industry in var.industries:
            optimal_taus[importer][industry] = result.x[count_idx * (var.num_industries) + idx]
            gov_obj_values[importer][industry] = -result.fun
            idx += 1
        count_idx += 1

    # Update the global `var.tau` with the new optimal tariffs
    for importer in optimal_taus:
        for industry in optimal_taus[importer]:
            var.tau[exporter_name][importer][industry] = optimal_taus[importer][industry]
    
    return optimal_taus, gov_obj_values

# Initialize a dictionary to store tariff values for each iteration
tariff_history = {exporter: {importer: {industry: [] for industry in var.industries} for importer in var.countries if importer != exporter} for exporter in var.countries}

# Ensure the directory exists
output_dir = "coop_img"
os.makedirs(output_dir, exist_ok=True)

# Initialize a dictionary to store welfare values for each iteration
welfare_history = {country: {industry: [] for industry in var.industries} for country in var.countries}

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
    return abs(total_welfare - cooperative_welfare_target)  # Absolute difference from the target

# Modify the constraints to include cooperative welfare
def constraints_with_cooperative_welfare(tau_js, j, cooperative_welfare_target):
    cons = constraints(tau_js, j)
    cons.append({'type': 'eq', 'fun': lambda tau_js, j=j: cooperative_obj(tau_js, cooperative_welfare_target, j)})
    return cons

# Run iterations until tariffs converge
for iteration in range(5):  # Arbitrary number of iterations
    previous_tau = {exporter: {importer: {industry: var.tau[exporter][importer][industry] for industry in var.industries} for importer in var.countries if importer != exporter} for exporter in var.countries}

    for exporter in var.countries:
        tau, gov_obj_values = calculate_optimum_tariffs(exporter)

        for importer in tau:
            for industry in tau[importer]:
                tariff_history[exporter][importer][industry].append(tau[importer][industry])

    # Plotting tariffs for the current iteration
    plt.figure()
    for importer in var.countries:
        if importer == exporter:
            continue
        for industry in var.industries:
            plt.plot(tariff_history[exporter][importer][industry], label=f"{importer} - {industry}")
    plt.xlabel("Iteration")
    plt.ylabel("Tariff")
    plt.legend()
    plt.title(f"Tariff Convergence for Exporter: {exporter}")
    plt.savefig(os.path.join(output_dir, f"tariff_convergence_{exporter}_iteration_{iteration}.png"))
    plt.close()

    # Update welfare history
    for country in var.countries:
        for industry in var.industries:
            welfare_history[country][industry].append(calc_welfare(country, industry))

    # Check for convergence (e.g., using a small threshold for changes in tariffs)
    converged = True
    for exporter in var.countries:
        for importer in var.countries:
            if importer == exporter:
                continue
            for industry in var.industries:
                if abs(previous_tau[exporter][importer][industry] - var.tau[exporter][importer][industry]) > 1e-5:
                    converged = False
                    break
            if not converged:
                break
        if not converged:
            break

    if converged:
        break

# After the loop ends, print the final cooperative tariffs:
print("Cooperative Tariffs (tau):")

for exporter in var.countries:
    print(f"\nTariffs for {exporter} as country_i:")
    
    # Create a DataFrame to organize the tariffs neatly
    df_tau = pd.DataFrame({importer: {industry: var.tau[exporter][importer][industry] 
                                      for industry in var.industries} 
                           for importer in var.countries if importer != exporter})

    # Print the DataFrame in the required format
    print(df_tau.T.to_string())

# Save tariff history to CSV
for exporter in var.countries:
    for importer in var.countries:
        if importer == exporter:
            continue
        for industry in var.industries:
            df = pd.DataFrame(tariff_history[exporter][importer][industry], columns=["Tariff"])
            df.to_csv(os.path.join(output_dir, f"tariff_history_{exporter}_{importer}_{industry}.csv"), index=False)

# Plotting welfare for each country
for country in var.countries:
    plt.figure()
    for industry in var.industries:
        plt.plot(welfare_history[country][industry], label=f"{industry}")
    plt.xlabel("Iteration")
    plt.ylabel("Welfare")
    plt.legend()
    plt.title(f"Welfare Convergence for Country: {country}")
    plt.savefig(os.path.join(output_dir, f"welfare_convergence_{country}.png"))
    plt.close()

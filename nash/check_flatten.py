import sys
import os
import matplotlib.pyplot as plt

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

var.fill_gamma()
var.fill_pi()
factual_pi = var.pi.copy() #factual var.pi 보존


var.fill_alpha()

def calc_x(j, s):
    sum = 0
    sum += var.w[j] * var.L_js[j][s] + var.pi[j][s] + var.L_js[j][s]/var.L_j[j] * var.alpha_denom[j][s]
    return sum


# Welfare function
def calc_welfare(j, s):
    return calc_x(j, s) + var.P_j[j]

def gov_obj(tau_js, exporter, importer):
    tau_copy = {i: {industry: 0 for industry in var.industries} for i in var.countries if i != importer}
    idx = 0
    for industry in var.industries:
        for country in var.countries:
            if country != importer:
                tau_copy[country][industry] = tau_js[idx]
                idx += 1
    total = 0
    for s in var.industries:
        total += var.pol_econ[importer][s] * calc_welfare(importer, s)
    return -total  # We minimize, so we return the negative

# Constraint 1 for country j and industry s
def eq_12(j, s):
    total = 0
    for i in var.countries:
        if i != j:
            total += (var.gamma[i][j][s] * (var.tau_hat[i][j][s] ** (1 - var.sigma[s]))) ** (1 / (1 - var.sigma[s]))
    var.P_hat[j][s] = total
    return total

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
                        (eq_12(j, s) ** (var.sigma[s] - 1)) * (abs(var.tau_hat[i][j][s]) ** -var.sigma[s]) + (var.pi[j][s] / x2(j) * var.pi_hat[j][s]))
    return total

def term3(j):
    total = 0
    for s in var.industries:
        for i in var.countries:
            if i != j:
                total += (var.pi[j][s] / x2(j) * var.pi_hat[j][s]) * var.alpha[j][i][s] * (abs(var.tau_hat[j][i][s]) ** -var.sigma[s]) * (var.w[i] ** (1 - var.sigma[s])) * (eq_12(i, s) ** (var.sigma[s] - 1))
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
    tau_copy = {i: {industry: 0 for industry in var.industries} for i in var.countries if i != j}
    idx = 0
    for industry in var.industries:
        for country in var.countries:
            if country != j:
                tau_copy[country][industry] = tau_js
                idx += 1
    cons = []
    #for s in var.industries:
        #cons.append({'type': 'eq', 'fun': lambda tau_js, j=j, s=s: eq_12(j, s)})
    cons.append({'type': 'eq', 'fun': lambda tau_js, j=j: eq_13(j)})
    for i in var.countries:
        for s in var.industries:
            cons.append({'type': 'eq', 'fun': lambda tau_js, i=i, s=s: eq_10(i, s)})
    return cons

# Function to generate tariff matrix
def generate_tariff_matrix():
    # Create an array of tariffs (excluding the home country)
    # Tariffs for each industry (rows) and country (columns, excluding the home country)
    tariff_values = np.random.rand(var.num_industries, var.num_countries - 1) * 0.5 + 1.0
    return tariff_values

# Generate an array of 5 tariff matrices
tariff_matrices = [generate_tariff_matrix() for _ in range(5)]

def flatten(matrix):
    return [item for sublist in matrix for item in sublist]
   
flat_matrices = [flatten(tariff_matrices[i]) for i in range(5)]

# Update hats function
def update_hats(tau, t, pi): #갱신된 값이 인자로 들어감
    global pi_hat, tau_hat, t_hat, factual_tau, factual_pi, factual_t
    for i in var.countries:
        for j in var.countries:
            if i != j:
                for s in var.industries:
                    var.tau_hat[i][j][s] = abs(tau[i][j][s] / var.factual_tau[i][j][s])
                    var.t_hat[i][j][s] = abs(t[i][j][s] / var.factual_t[i][j][s])
    for j in var.countries:
        for s in var.industries:
            var.pi_hat[j][s] = abs(var.pi[j][s] / factual_pi[j][s])


def calculate_optimum_tariffs(exporter, importer, initial_guess):
    result = minimize(gov_obj, initial_guess, args=(exporter, importer), constraints=constraints(initial_guess, importer))
    optimal_taus = {industry: result.x[idx] for idx, industry in enumerate(var.industries)}
    return optimal_taus

# Initialize a dictionary to store tariff values for each iteration
tariff_history = {i: {j: {industry: [] for industry in var.industries} for j in var.countries if j != i} for i in var.countries}

# Ensure the directory exists
output_dir = "0804_nash_img"
os.makedirs(output_dir, exist_ok=True)


# Main iteration loop
for iteration in range(100):
    print(f"Iteration {iteration + 1}")
    
    new_taus = {i: {j: {industry: 0 for industry in var.industries} for j in var.countries if j != i} for i in var.countries}
    tariff_matrices = [generate_tariff_matrix() for _ in range(len(var.countries))]
    flat_matrices = [flatten(tariff_matrices[i]) for i in range(len(var.countries))]
    
    for k, exporter in enumerate(var.countries):
        for importer in var.countries:
            if exporter != importer:
                initial_guess = flatten(new_taus[exporter][importer])
                optimal_taus_for_pair= calculate_optimum_tariffs(exporter, importer, initial_guess)
                for industry in var.industries:
                    new_taus[exporter][importer][industry] = optimal_taus_for_pair[industry]
    
    # Update var.tau with new_taus
    for exporter in var.countries:
        for importer in var.countries:
            if importer != exporter:
                for industry in var.industries:
                    var.tau[exporter][importer][industry] = new_taus[exporter][importer][industry]
                    tariff_history[exporter][importer][industry].append(var.tau[exporter][importer][industry])  # Store the tariff value

    # Print the current state of var.tau
    print("Nash Tariffs (tau) after iteration", iteration + 1)
    for i in var.countries:
        print(f"\nTariffs for {i} as the home country:")
        df_tau = pd.DataFrame({j: {s: var.tau[i][j][s] for s in var.industries} for j in var.countries if j != i})
        print(df_tau)

    # Recalculate gamma, var.pi, and alpha with new tau values
    update_hats(var.tau, var.t, var.pi)

# Plot and save the tariff history for each combination of exporter, importer, and industry
iterations = list(range(1, 101))

for exporter in var.countries:
    for importer in var.countries:
        if exporter != importer:
            for industry in var.industries:
                tariffs = tariff_history[exporter][importer][industry]

                plt.figure(figsize=(10, 6))
                plt.plot(iterations, tariffs, marker='o', color='r')
                plt.title(f'Tariff for "{industry}" from {exporter} to {importer} in Repeated Game')
                plt.xlabel('Iteration')
                plt.ylabel('Tariff')
                plt.grid(True)
                
                # Save the plot
                file_name = f"{output_dir}/tariff_{industry}_{exporter}_to_{importer}.png"
                plt.savefig(file_name)
                plt.close()
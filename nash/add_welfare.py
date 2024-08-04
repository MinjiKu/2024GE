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

def gov_obj(tau_js, j):
    tau_copy = {i: {industry: 0 for industry in var.industries} for i in var.countries if i != j}
    idx = 0
    for industry in var.industries:
        for country in var.countries:
            if country != j:
                tau_copy[country][industry] = tau_js[idx]
                idx += 1
    # Rest of the function remains the same
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
                if idx >= len(tau_js):  # Check if index is within bounds
                    print(f"IndexError: idx={idx} out of bounds for tau_js with length {len(tau_js)}")
                    return []
                tau_copy[country][industry] = tau_js[idx]
                idx += 1

    # Print for debugging
    print(f"tau_copy: {tau_copy}")            
    
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


# Function to generate tariff matrix
def generate_tariff_matrix():
    # Create an array of tariffs (excluding the home country)
    # Tariffs for each industry (rows) and country (columns, excluding the home country)
    tariff_values = np.random.rand(var.num_industries, var.num_countries - 1) * 0.5 + 1.0
    return tariff_values

def flatten(matrix):
    return [item for sublist in matrix for item in sublist]

# Generate an array of 5 tariff matrices
tariff_matrices = [generate_tariff_matrix() for _ in range(5)]   
flat_matrices = [flatten(tariff_matrices[i]) for i in range(5)]

def generate_tariff_matrix():
    tariff_values = np.random.rand(var.num_industries, var.num_countries - 1) * 0.5 + 1.0
    return tariff_values

def cal_x_j(country):
    sum = 0
    for industry in var.industries:
        sum += var.x[country][industry]
    return sum

def cal_delta_p_js(country, industry):
    sum = 0
    for c in var.countries:
        if c == country: continue
        sum += delta_p[c][country][industry]
    return sum

def cal_p_js(country, industry):
    sum = 0
    for c in var.countries:
        if c == country: continue
        sum += var.p_is[c][country][industry]
    return sum

def cal_delta_p_is(country, industry):
    sum = 0
    for c in var.countries:
        if c == country: continue
        sum += delta_p[country][c][industry]
    return sum

def cal_p_is(country, industry):
    sum = 0
    for c in var.countries:
        if c == country: continue
        sum += var.p_is[country][c][industry]
    return sum

def welfare_change(T, X, delta_p, p, pi, t, delta_pi, delta_T):
    delta_W_W = {}

    x_js_list = {country: {industry: 0 for industry in var.industries} for country in var.countries}
    x_j_list = {country: 0 for country in var.countries}
    
    for j in var.countries:
        for s in var.industries:
            x_js_list[j][s] = calc_x(j, s)
            x_j_list[j] += x_js_list[j][s]
            
    
    for j in var.countries:  # j국 (수입국)
        term1 = 0
        term2 = 0
        term3 = 0
        
        for i in var.countries:  # i국 (수출국)
            if i != j:
                for s in var.industries:  # s산업
                    term1 += (T[i][j][s] / cal_x_j(j)) * ((cal_delta_p_js(j, s) / cal_p_js(j,s)) - (cal_delta_p_is(i,s) / cal_p_is(i,s)))
                    term3 += (t[i][j][s] * T[i][j][s] / cal_x_j(j)) * ((delta_T[i][j][s] / T[i][j][s]) - (cal_delta_p_is(i,s) / cal_p_is(i,s)))
        
        for s in var.industries:  # s산업
            term2 += (pi[j][s] / x_j_list[j]) * ((delta_pi[j][s] / pi[j][s]) - (cal_delta_p_js(j, s) / cal_p_js(j,s)))
        
        delta_W_W[j] = term1 + term2 + term3
    
    return delta_W_W



# Update hats function
def update_hats(tau, t, pi): #갱신된 값이 인자로 들어감
    # global pi_hat, tau_hat, t_hat, factual_tau, factual_pi, factual_t
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
    global tau, t, tariff_matrices;  # We'll modify both the global tau and t variables
    
    optimal_taus = {j: {industry: 0 for industry in var.industries} for j in var.countries if j != exporter_name}
    
    for j, importer in enumerate(var.countries):
        if importer == exporter_name:
            continue
        
        idx = 0

        for k, industry in enumerate(var.industries):
            
            # result = minimize(gov_obj, flat_matrices[j], args=(importer,), constraints=constraints(tariff_matrices[j], importer))
            # optimal_taus[importer][industry] = flat_matrices[j][k * (var.num_countries - 1)+ idx]
            result = minimize(gov_obj, flat_matrices[j], args=(importer,), constraints=constraints(flat_matrices[j], importer))
            optimal_taus[importer][industry] = result.x[idx]
            idx += 1
            
            var.tau[exporter_name] = optimal_taus
            var.fill_gamma()
            
    
    # welfare_change(var.T, var.x, delta_p, var.p_is, var.pi, var.t, delta_pi, delta_T)
    return optimal_taus


# 임시 딕셔너리 생성
temp_pi = var.pi.copy()
temp_p = var.p_is.copy()
temp_t = var.t.copy()
temp_T = var.T.copy()

# Perform 100 iterations
for iteration in range(1):
    print(f"Iteration {iteration + 1}") 
    
    new_taus = {i: {j: {industry: 0 for industry in var.industries} for j in var.countries if j != i} for i in var.countries}
    tariff_matrices = [generate_tariff_matrix() for _ in range(5)]
    flat_matrices = [flatten(tariff_matrices[i]) for i in range(5)]
    
    for k, country in enumerate(var.countries):
        new_taus[country] = calculate_optimum_tariffs(country)

    # Print the final Nash tariffs and corresponding t values
    print("Nash Tariffs (tau):")
    for i in var.countries:
        print(f"\nTariffs for {i} as the home country:")
        df_tau = pd.DataFrame({j: {s: new_taus[i][j][s] for s in var.industries} for j in var.countries if j != i})
        print(df_tau)

    temp_t = var.t.copy()
    temp_pi = var.pi.copy()

    # Update tau with new values and adjust t accordingly
    for i in var.countries:
        for j in var.countries:
            if j != i:
                for industry in var.industries:
                    var.tau[i][j][industry] = new_taus[i][j][industry]
                    new_t = var.tau[i][j][industry] - 1
                    var.t[i][j][industry] = max(new_t, 1e-10)  # Ensure t is not below 1e-10
    
    # Recalculate gamma, var.pi, and alpha with new tau values
    update_hats(var.tau, var.t, var.pi)
    
    # Update temp variables for next iteration 
    temp_p = var.p_is.copy() 
    temp_T = var.T.copy()

    # Delta 값 계산
    delta_pi = {country: {industry: var.pi[country][industry] - temp_pi[country][industry] for industry in var.industries} for country in var.countries}
    delta_p = {i: {j: {industry: var.p_is[i][j][industry] - temp_p[i][j][industry] for industry in var.industries} for j in var.countries if i != j} for i in var.countries}
    delta_T = {i: {j: {industry: var.T[i][j][industry] - temp_T[i][j][industry] for industry in var.industries} for j in var.countries if i != j} for i in var.countries}

    # Call welfare_change with updated delta values
    welfare_change(var.T, var.x, delta_p, var.p_is, var.pi, var.t, delta_pi, delta_T)
    print("welfare change: ")
    print(welfare_change(var.T, var.x, delta_p, var.p_is, var.pi, var.t, delta_pi, delta_T))
    print("\n")

# Print the final Nash tariffs and corresponding t values
print("Nash Tariffs (tau):")
for i in var.countries:
    print(f"\nTariffs for {i} as the home country:")
    df_tau = pd.DataFrame({j: {s: var.tau[i][j][s] for s in var.industries} for j in var.countries if j != i})
    print(df_tau)

print("\nCorresponding t values:")
for i in var.countries:
    print(f"\nt values for {i} as the home country:")
    df_t = pd.DataFrame({j: {s: var.t[i][j][s] for s in var.industries} for j in var.countries if j != i})
    print(df_t)

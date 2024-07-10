# # change current array data structures to dictionary structure

# this version has zero division error 

# import numpy as np
# from scipy.optimize import minimize
# import pandas as pd

# # Define countries and industries
# countries = ['China', 'Korea', 'Japan', 'US', 'Germany']
# industries = ['Gim', 'Steel', 'Semiconductor', 'Car']
# num_countries = len(countries)
# num_industries = len(industries)

# # Initialize dictionaries
# pol_econ = {country: {industry: 0 for industry in industries} for country in countries}
# x = {country: {industry: 0 for industry in industries} for country in countries}
# p = {country: 0 for country in countries}
# gamma = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}
# tau = {i: {j: {industry: 1 for industry in industries} for j in countries} for i in countries}
# sigma = {industry: 1 for industry in industries}
# t = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}
# T = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}
# pi = {country: {industry: 0 for industry in industries} for country in countries}
# pi_hat = {country: {industry: 0 for industry in industries} for country in countries}
# alpha = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}
# tau_hat = {i: {j: {industry: 1 for industry in industries} for j in countries} for i in countries}
# t_hat = {i: {j: {industry: 1 for industry in industries} for j in countries} for i in countries}
# w_hat = {country: 1 for country in countries}

# # Example values (you can fill with actual data or random values)
# for country in countries:
#     for industry in industries:
#         pol_econ[country][industry] = np.random.rand()
#         x[country][industry] = np.random.rand()
#     p[country] = np.random.rand()

# for i in countries:
#     for j in countries:
#         for industry in industries:
#             gamma[i][j][industry] = np.random.rand()
#             tau[i][j][industry] = np.random.rand() + 1
#             t[i][j][industry] = np.random.rand()
#             T[i][j][industry] = np.random.rand()
#             alpha[i][j][industry] = np.random.rand()
#             tau_hat[i][j][industry] = np.random.rand() + 1
#             t_hat[i][j][industry] = np.random.rand() + 1

# for country in countries:
#     for industry in industries:
#         pi[country][industry] = np.random.rand()
#         pi_hat[country][industry] = np.random.rand()

# # Welfare function
# def welfare(j, s):
#     return x[j][s] + p[j]

# # Government objective function for country j
# def gov_obj(tau_js, j):
#     tau_copy = tau.copy()
#     for i, industry in enumerate(industries):
#         for k, country in enumerate(countries):
#             tau_copy[country][j][industry] = tau_js[i * num_countries + k]
#     total = 0
#     for s in industries:
#         total += pol_econ[j][s] * welfare(j, s)
#     return -total  # We minimize, so we return the negative

# # Constraint 1 for country j and industry s
# def eq_12(j, s):
#     total = 0
#     for i in countries:
#         total += (gamma[i][j][s] * (tau[i][j][s] ** (1 - sigma[s]))) ** (1 / (1 - sigma[s]))
#     return total

# # Constraint 2 helper functions
# def x2(j):
#     total = 0
#     for i in countries:
#         for s in industries:
#             total += t[i][j][s] * T[i][j][s]
#     return total

# def wL(j):
#     term2 = 0
#     for i in countries:
#         for s in industries:
#             term2 += t[i][j][s] * T[i][j][s]

#     term3 = 0
#     for s in industries:
#         term3 += pi[j][s]

#     return x2(j) - term2 - term3

# def x2_hat(j):
#     return 1  # Simplified

# def complicated(j):
#     total = 0
#     for i in countries:
#         for s in industries:
#             total += (t[i][j][s] * T[i][j][s] / x2(j) * t_hat[i][j][s] * 
#                       (eq_12(j, s) ** (sigma[s] - 1)) * (tau_hat[i][j][s] ** -sigma[s]) * 
#                       x2_hat(j) + (pi[j][s] / x2(j) * pi_hat[j][s]))
#     return total

# def eq_13(j):
#     term1 = wL(j) / x2(j)
#     term2 = complicated(j)
#     term3 = 0

#     for s in industries:
#         term3 += pi[j][s] / x2(j) * pi_hat[j][s]

#     return term1 + term2 + term3

# # Constraint 3 for country i and industry s
# def eq_10(i, s):
#     total = 0
#     for j in countries:
#         total += (alpha[i][j][s] * (tau_hat[i][j][s] ** -sigma[s]) * 
#                   (w_hat[i] ** (1 - sigma[s])) * (eq_12(j, s) ** (sigma[s] - 1)) * eq_13(j))
#     return total

# # Constraints as a list for country j
# def constraints(tau_js, j):
#     tau_copy = tau.copy()
#     for i, industry in enumerate(industries):
#         for k, country in enumerate(countries):
#             tau_copy[country][j][industry] = tau_js[i * num_countries + k]
#     cons = []
#     for s in industries:
#         cons.append({'type': 'eq', 'fun': lambda tau_js, j=j, s=s: eq_12(j, s) - 1})
#     cons.append({'type': 'eq', 'fun': lambda tau_js, j=j: eq_13(j) - 1})
#     for i in countries:
#         for s in industries:
#             cons.append({'type': 'eq', 'fun': lambda tau_js, i=i, s=s: eq_10(i, s) - 1})
#     return cons

# # Optimize tariffs for each country j
# optimal_taus = {country: {industry: 0 for industry in industries} for country in countries}
# for j in countries:
#     initial_tau_js = np.ones((num_countries, num_industries)).flatten()
#     result = minimize(gov_obj, initial_tau_js, args=(j,), constraints=constraints(initial_tau_js, j))
#     for i, industry in enumerate(industries):
#         for k, country in enumerate(countries):
#             optimal_taus[country][industry] = result.x[i * num_countries + k]

# print("Optimal tariffs for each country in all industries:")
# print(pd.DataFrame(optimal_taus))

# # this version has only one table
# import numpy as np
# from scipy.optimize import minimize
# import pandas as pd

# # Define countries and industries
# countries = ['China', 'Korea', 'Japan', 'US', 'Germany']
# industries = ['Gim', 'Steel', 'Semiconductor', 'Car']
# num_countries = len(countries)
# num_industries = len(industries)

# # Initialize dictionaries
# pol_econ = {country: {industry: 0 for industry in industries} for country in countries}
# x = {country: {industry: 0 for industry in industries} for country in countries}
# p = {country: 0 for country in countries}
# gamma = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}
# tau = {i: {j: {industry: 1 for industry in industries} for j in countries} for i in countries}
# sigma = {industry: 1.1 for industry in industries}  # Avoid sigma[s] = 1 to prevent division by zero
# t = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}
# T = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}
# pi = {country: {industry: 0 for industry in industries} for country in countries}
# pi_hat = {country: {industry: 0 for industry in industries} for country in countries}
# alpha = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}
# tau_hat = {i: {j: {industry: 1 for industry in industries} for j in countries} for i in countries}
# t_hat = {i: {j: {industry: 1 for industry in industries} for j in countries} for i in countries}
# w_hat = {country: 1 for country in countries}

# # Example values (you can fill with actual data or random values)
# for country in countries:
#     for industry in industries:
#         pol_econ[country][industry] = np.random.rand()
#         x[country][industry] = np.random.rand()
#     p[country] = np.random.rand()

# for i in countries:
#     for j in countries:
#         for industry in industries:
#             gamma[i][j][industry] = np.random.rand()
#             tau[i][j][industry] = np.random.rand() + 1
#             t[i][j][industry] = np.random.rand()
#             T[i][j][industry] = np.random.rand()
#             alpha[i][j][industry] = np.random.rand()
#             tau_hat[i][j][industry] = np.random.rand() + 1
#             t_hat[i][j][industry] = np.random.rand() + 1

# for country in countries:
#     for industry in industries:
#         pi[country][industry] = np.random.rand()
#         pi_hat[country][industry] = np.random.rand()

# # Welfare function
# def welfare(j, s):
#     return x[j][s] + p[j]

# # Government objective function for country j
# def gov_obj(tau_js, j):
#     tau_copy = tau.copy()
#     for i, industry in enumerate(industries):
#         for k, country in enumerate(countries):
#             tau_copy[country][j][industry] = tau_js[i * num_countries + k]
#     total = 0
#     for s in industries:
#         total += pol_econ[j][s] * welfare(j, s)
#     return -total  # We minimize, so we return the negative

# # Constraint 1 for country j and industry s
# def eq_12(j, s):
#     total = 0
#     for i in countries:
#         total += (gamma[i][j][s] * (tau[i][j][s] ** (1 - sigma[s]))) ** (1 / (1 - sigma[s]))
#     return total

# # Constraint 2 helper functions
# def x2(j):
#     total = 0
#     for i in countries:
#         for s in industries:
#             total += t[i][j][s] * T[i][j][s]
#     return total

# def wL(j):
#     term2 = 0
#     for i in countries:
#         for s in industries:
#             term2 += t[i][j][s] * T[i][j][s]

#     term3 = 0
#     for s in industries:
#         term3 += pi[j][s]

#     return x2(j) - term2 - term3

# def x2_hat(j):
#     return 1  # Simplified

# def complicated(j):
#     total = 0
#     for i in countries:
#         for s in industries:
#             total += (t[i][j][s] * T[i][j][s] / x2(j) * t_hat[i][j][s] * 
#                       (eq_12(j, s) ** (sigma[s] - 1)) * (tau_hat[i][j][s] ** -sigma[s]) * 
#                       x2_hat(j) + (pi[j][s] / x2(j) * pi_hat[j][s]))
#     return total

# def eq_13(j):
#     term1 = wL(j) / x2(j)
#     term2 = complicated(j)
#     term3 = 0

#     for s in industries:
#         term3 += pi[j][s] / x2(j) * pi_hat[j][s]

#     return term1 + term2 + term3

# # Constraint 3 for country i and industry s
# def eq_10(i, s):
#     total = 0
#     for j in countries:
#         total += (alpha[i][j][s] * (tau_hat[i][j][s] ** -sigma[s]) * 
#                   (w_hat[i] ** (1 - sigma[s])) * (eq_12(j, s) ** (sigma[s] - 1)) * eq_13(j))
#     return total

# # Constraints as a list for country j
# def constraints(tau_js, j):
#     tau_copy = tau.copy()
#     for i, industry in enumerate(industries):
#         for k, country in enumerate(countries):
#             tau_copy[country][j][industry] = tau_js[i * num_countries + k]
#     cons = []
#     for s in industries:
#         cons.append({'type': 'eq', 'fun': lambda tau_js, j=j, s=s: eq_12(j, s) - 1})
#     cons.append({'type': 'eq', 'fun': lambda tau_js, j=j: eq_13(j) - 1})
#     for i in countries:
#         for s in industries:
#             cons.append({'type': 'eq', 'fun': lambda tau_js, i=i, s=s: eq_10(i, s) - 1})
#     return cons

# # Optimize tariffs for each country j
# optimal_taus = {country: {industry: 0 for industry in industries} for country in countries}
# for j in countries:
#     initial_tau_js = np.ones((num_countries, num_industries)).flatten()
#     result = minimize(gov_obj, initial_tau_js, args=(j,), constraints=constraints(initial_tau_js, j))
#     for i, industry in enumerate(industries):
#         for k, country in enumerate(countries):
#             optimal_taus[country][industry] = result.x[i * num_countries + k]

# print("Optimal tariffs for each country in all industries:")
# print(pd.DataFrame(optimal_taus))

import numpy as np
from scipy.optimize import minimize
import pandas as pd

# Define countries and industries
countries = ['China', 'Korea', 'Japan', 'US', 'Germany']
industries = ['Gim', 'Steel', 'Semiconductor', 'Car']
num_countries = len(countries)
num_industries = len(industries)

# Initialize dictionaries
pol_econ = {country: {industry: 0 for industry in industries} for country in countries}
x = {country: {industry: 0 for industry in industries} for country in countries}
p = {country: 0 for country in countries}
gamma = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}
tau = {i: {j: {industry: 1 for industry in industries} for j in countries} for i in countries}
sigma = {industry: 1.1 for industry in industries}  # Avoid sigma[s] = 1 to prevent division by zero
t = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}
T = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}
pi = {country: {industry: 0 for industry in industries} for country in countries}
pi_hat = {country: {industry: 0 for industry in industries} for country in countries}
alpha = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}
tau_hat = {i: {j: {industry: 1 for industry in industries} for j in countries} for i in countries}
t_hat = {i: {j: {industry: 1 for industry in industries} for j in countries} for i in countries}
w_hat = {country: 1 for country in countries}

# Example values (you can fill with actual data or random values)
for country in countries:
    for industry in industries:
        pol_econ[country][industry] = np.random.rand()
        x[country][industry] = np.random.rand()
    p[country] = np.random.rand()

for i in countries:
    for j in countries:
        for industry in industries:
            gamma[i][j][industry] = np.random.rand()
            tau[i][j][industry] = np.random.rand() + 1
            t[i][j][industry] = np.random.rand()
            T[i][j][industry] = np.random.rand()
            alpha[i][j][industry] = np.random.rand()
            tau_hat[i][j][industry] = np.random.rand() + 1
            t_hat[i][j][industry] = np.random.rand() + 1

for country in countries:
    for industry in industries:
        pi[country][industry] = np.random.rand()
        pi_hat[country][industry] = np.random.rand()

# Welfare function
def welfare(j, s):
    return x[j][s] + p[j]

# Government objective function for country j
def gov_obj(tau_js, j):
    tau_copy = tau.copy()
    for i, industry in enumerate(industries):
        for k, country in enumerate(countries):
            tau_copy[country][j][industry] = tau_js[i * num_countries + k]
    total = 0
    for s in industries:
        total += pol_econ[j][s] * welfare(j, s)
    return -total  # We minimize, so we return the negative

# Constraint 1 for country j and industry s
def eq_12(j, s):
    total = 0
    for i in countries:
        total += (gamma[i][j][s] * (tau[i][j][s] ** (1 - sigma[s]))) ** (1 / (1 - sigma[s]))
    return total

# Constraint 2 helper functions
def x2(j):
    total = 0
    for i in countries:
        for s in industries:
            total += t[i][j][s] * T[i][j][s]
    return total

def wL(j):
    term2 = 0
    for i in countries:
        for s in industries:
            term2 += t[i][j][s] * T[i][j][s]

    term3 = 0
    for s in industries:
        term3 += pi[j][s]

    return x2(j) - term2 - term3

def x2_hat(j):
    return 1  # Simplified

def complicated(j):
    total = 0
    for i in countries:
        for s in industries:
            total += (t[i][j][s] * T[i][j][s] / x2(j) * t_hat[i][j][s] * 
                      (eq_12(j, s) ** (sigma[s] - 1)) * (tau_hat[i][j][s] ** -sigma[s]) * 
                      x2_hat(j) + (pi[j][s] / x2(j) * pi_hat[j][s]))
    return total

def eq_13(j):
    term1 = wL(j) / x2(j)
    term2 = complicated(j)
    term3 = 0

    for s in industries:
        term3 += pi[j][s] / x2(j) * pi_hat[j][s]

    return term1 + term2 + term3

# Constraint 3 for country i and industry s
def eq_10(i, s):
    total = 0
    for j in countries:
        total += (alpha[i][j][s] * (tau_hat[i][j][s] ** -sigma[s]) * 
                  (w_hat[i] ** (1 - sigma[s])) * (eq_12(j, s) ** (sigma[s] - 1)) * eq_13(j))
    return total

# Constraints as a list for country j
def constraints(tau_js, j):
    tau_copy = tau.copy()
    for i, industry in enumerate(industries):
        for k, country in enumerate(countries):
            tau_copy[country][j][industry] = tau_js[i * num_countries + k]
    cons = []
    for s in industries:
        cons.append({'type': 'eq', 'fun': lambda tau_js, j=j, s=s: eq_12(j, s) - 1})
    cons.append({'type': 'eq', 'fun': lambda tau_js, j=j: eq_13(j) - 1})
    for i in countries:
        for s in industries:
            cons.append({'type': 'eq', 'fun': lambda tau_js, i=i, s=s: eq_10(i, s) - 1})
    return cons

# Optimize tariffs for each country i (home country)
for i in countries:
    optimal_taus = {j: {industry: 0 for industry in industries} for j in countries if j != i}
    for j in countries:
        if j == i:
            continue
        initial_tau_js = np.ones((num_countries, num_industries)).flatten()
        result = minimize(gov_obj, initial_tau_js, args=(j,), constraints=constraints(initial_tau_js, j))
        for k, industry in enumerate(industries):
            for m, country in enumerate(countries):
                if country != i:
                    optimal_taus[country][industry] = result.x[k * num_countries + m]
    
    print(f"Optimal tariffs for {i} as the home country:")
    print(pd.DataFrame(optimal_taus))
    print("\n")

import numpy as np
from scipy.optimize import minimize
import pandas as pd

# Define countries and industries
countries = ['China', 'Korea', 'Japan', 'USA', 'Germany']
industries = ['gim', 'steel', 'semi', 'car']
num_countries = len(countries)
num_industries = len(industries)

# Initialize dictionaries
pol_econ = {country: {industry: 0 for industry in industries} for country in countries}
x = {country: {industry: 0 for industry in industries} for country in countries}
p = {country: 0 for country in countries}
gamma = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}
tau = {i: {j: {industry: 1 for industry in industries} for j in countries} for i in countries}

# Elasticities
sigma = {'gim': 3.57, 'steel': 4.0, 'semi': 2.5, 'car': 1.8}
# sigma = {industry: 1.1 for industry in industries}  # Avoid sigma[s] = 1 to prevent division by zero

t = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}

# T = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}
# Trade flows data structure
T = {
    'Korea': {'USA': {'gim': 196766, 'steel': 580234, 'semi': 711206, 'car': 72066}, 
              'China': {'gim': 101299, 'steel': 454598, 'semi': 17018970, 'car': 16041614},
              'Japan': {'gim': 187401, 'steel': 903090, 'semi': 308979, 'car': 12433}, 
              'Germany': {'gim': 12194, 'steel': 53993, 'semi': 121601, 'car': 657654}},
    'USA': {'Korea': {'gim': 19076, 'steel': 95, 'semi': 1092079, 'car': 963233}, 
            'China': {'gim': 15075, 'steel': 820, 'semi': 8306170, 'car': 9030967},
            'Japan': {'gim': 20000, 'steel': 5000, 'semi': 15000, 'car': 70000}, 
            'Germany': {'gim': 23529, 'steel': 8000, 'semi': 25000, 'car': 90000}},
    'China': {'Korea': {'gim': 28052, 'steel': 1803944, 'semi': 17758308, 'car': 1212067}, 
              'USA': {'gim': 113463, 'steel': 1325, 'semi': 2324998, 'car': 2610238},
              'Japan': {'gim': 306966, 'steel': 83818, 'semi': 3119372, 'car': 2610238}, 
              'Germany': {'gim': 23529, 'steel': 582, 'semi': 1636760, 'car': 1833607}},
    'Japan': {'Korea': {'gim': 1329, 'steel': 2055349, 'semi': 1921372, 'car': 296440}, 
              'USA': {'gim': 5000, 'steel': 1000, 'semi': 12000, 'car': 80000},
              'China': {'gim': 1459, 'steel': 777891, 'semi': 6882555, 'car': 6714475}, 
              'Germany': {'gim': 738, 'steel': 12493, 'semi': 370643, 'car': 2518119}},
    'Germany': {'Korea': {'gim': 125, 'steel': 681, 'semi': 628437, 'car': 6468871}, 
                'USA': {'gim': 10000, 'steel': 5000, 'semi': 30000, 'car': 120000},
                'Japan': {'gim': 110, 'steel': 40, 'semi': 434221, 'car': 4282380}, 
                'China': {'gim': 122, 'steel': 5201, 'semi': 1436904, 'car': 16512164}}
}

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
            # T[i][j][industry] = np.random.rand()
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
            if j != country:
                tau_copy[country][j][industry] = tau_js[i * num_countries + k]
    total = 0
    for s in industries:
        total += pol_econ[j][s] * welfare(j, s)
    return -total  # We minimize, so we return the negative

# Constraint 1 for country j and industry s
def eq_12(j, s):
    total = 0
    for i in countries:
        if i != j:
            total += (gamma[i][j][s] * (tau[i][j][s] ** (1 - sigma[s]))) ** (1 / (1 - sigma[s]))
    return total

# Constraint 2 helper functions
def x2(j):
    total = 0
    for i in countries:
        for s in industries:
            if i != j:
                total += t[i][j][s] * T[i][j][s]
    return total

def wL(j):
    term2 = 0
    for i in countries:
        for s in industries:
            if i != j:
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
            if i != j:
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
        if i != j:
            total += (alpha[i][j][s] * (tau_hat[i][j][s] ** -sigma[s]) * 
                    (w_hat[i] ** (1 - sigma[s])) * (eq_12(j, s) ** (sigma[s] - 1)) * eq_13(j))
    return total

# Constraints as a list for country j
def constraints(tau_js, j):
    tau_copy = tau.copy()
    for i, industry in enumerate(industries):
        for k, country in enumerate(countries):
            if country != j:
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
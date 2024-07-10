import numpy as np
from scipy.optimize import minimize

# Define countries and industries
countries = ['China', 'Korea', 'Japan', 'US', 'Germany']
industries = ['Gim', 'Steel', 'Semiconductor', 'Car']
num_countries = len(countries)
num_industries = len(industries)

# political economy weights
pol_econ = [[0 for s in range(num_industries)] for j in range(num_countries)]

# x_js = the nominal income in industry s of country j
x = [[0 * num_industries] * num_countries]

# P_j = consumer price index (CPI)
p = [0 * num_countries]

# gamma = ??
gamma = [[[0 for s in range(num_industries)] for j in range(num_countries)] for i in range(num_countries)]

# tau = tax rate + 1
tau = [[[0 for s in range(num_industries)] for j in range(num_countries)] for i in range(num_countries)]

# sigma = ??
sigma = [0 for s in range(num_industries)]

t = [[[0 * num_industries] * num_countries] * num_countries]

T = [[[0 * num_industries] * num_countries] * num_countries]

pi = [[0 * num_industries] * num_countries]

# welfare function
def welfare(j, s):
    sum = 0
    sum += x[j][s] + p[j]
    return sum

# government objective function
def gov_obj(j):
    sum = 0
    for s in industries:
        sum += pol_econ[j][s] + welfare(j, s)

    return sum

# constraint 1
# needs to be modified to behave as a constraint
def eq_12(j, s):
    sum = 0
    for i in countries:
        sum += (gamma[i][j][s] * (tau[i][j][s] ** (1-sigma[s]))) ** ( 1 / (1 - sigma[s]))

def wL(j):
    term2 = 0
    for i in countries:
        for s in industries:
            term2 += t[i][j][s] * T[i][j][s]

    term3 = 0
    for s in industries:
        term3 += pi[j][s]

    res = x2[j] - term2 - term3
    return res

def x2(j):
    term = 0
    for i in countries:
        for s in industries:
            term += t[i][j][s] * T[i][j][s]
    return term


# constraint 2
def eq_13(j):
    term1 = wL(j) / x2(j) 
    
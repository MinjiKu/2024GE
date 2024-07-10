# import numpy as np
# from scipy.optimize import minimize

# # Define countries and industries
# countries = ['China', 'Korea', 'Japan', 'US', 'Germany']
# industries = ['Gim', 'Steel', 'Semiconductor', 'Car']
# num_countries = len(countries)
# num_industries = len(industries)

# # political economy weights
# pol_econ = [[0 for s in range(num_industries)] for j in range(num_countries)]

# # x_js = the nominal income in industry s of country j
# x = [[0 * num_industries] * num_countries]

# # P_j = consumer price index (CPI)
# p = [0 * num_countries]

# # gamma = ??
# gamma = [[[0 for s in range(num_industries)] for j in range(num_countries)] for i in range(num_countries)]

# # tau = tax rate + 1
# tau = [[[0 for s in range(num_industries)] for j in range(num_countries)] for i in range(num_countries)]

# # sigma = ??
# sigma = [0 for s in range(num_industries)]

# t = [[[0 * num_industries] * num_countries] * num_countries]

# T = [[[0 * num_industries] * num_countries] * num_countries]

# pi = [[0 * num_industries] * num_countries]

# pi_hat = [[0 * num_industries] * num_countries]

# alpha = [[[0 * num_industries] * num_countries] * num_countries]

# tau_hat = [[[0 * num_industries] * num_countries] * num_countries]

# t_hat = [[[0 * num_industries] * num_countries] * num_countries]

# w_hat = [0 * num_countries]

# # welfare function
# def welfare(j, s):
#     sum = 0
#     sum += x[j][s] + p[j]
#     return sum

# # government objective function
# def gov_obj(j):
#     sum = 0
#     for s in industries:
#         sum += pol_econ[j][s] + welfare(j, s)

#     return sum

# # constraint 1
# # needs to be modified to behave as a constraint
# def eq_12(j, s):
#     sum = 0
#     for i in countries:
#         sum += (gamma[i][j][s] * (tau[i][j][s] ** (1-sigma[s]))) ** ( 1 / (1 - sigma[s]))
#     return sum

# def wL(j):
#     term2 = 0
#     for i in countries:
#         for s in industries:
#             term2 += t[i][j][s] * T[i][j][s]

#     term3 = 0
#     for s in industries:
#         term3 += pi[j][s]

#     res = x2[j] - term2 - term3
#     return res

# def x2(j):
#     term = 0
#     for i in countries:
#         for s in industries:
#             term += t[i][j][s] * T[i][j][s]
#     return term

# def x2_hat(j):
#     # to be continued.
#     return 1

# def complicated(j):
#     res = 0
#     for i in countries:
#         for s in industries:
#             # needs to be modified
#             res += t[i][j][s] * T[i][j][s] / x2(j) * t_hat[i][j][s] * (eq_12(j, s) ** (sigma[s] - 1)) * (tau_hat[i][j][s] ** -sigma[s]) * x2_hat(j) + (pi[j][s]/ x2(j) * pi_hat[j][s])
#     return res

# # constraint 2
# def eq_13(j):
#     term1 = wL(j) / x2(j) 
#     term2 = complicated(j)
#     term3 = 0

#     for s in industries:
#         term3 += pi[j][s] / x2(j) * pi_hat[j][s]

#     return term1 + term2 + term3

# #constraint 3
# def eq_10(i, s):
#     res = 0
#     for j in countries:
#         res += alpha[i][j][s] * (tau_hat[i][j][s] ** -sigma[s]) * (w_hat[i] ** (1 - sigma[s])) * (eq_12(j, s) ** (sigma[s]-1)) * eq_13(j) 
#     return res


import numpy as np
from scipy.optimize import minimize

# Define countries and industries
countries = ['China', 'Korea', 'Japan', 'US', 'Germany']
industries = ['Gim', 'Steel', 'Semiconductor', 'Car']
num_countries = len(countries)
num_industries = len(industries)

# Initialize arrays
pol_econ = np.zeros((num_countries, num_industries))
x = np.zeros((num_countries, num_industries))
p = np.zeros(num_countries)
gamma = np.zeros((num_countries, num_countries, num_industries))
tau = np.ones((num_countries, num_countries, num_industries))
sigma = np.ones(num_industries)
t = np.ones((num_countries, num_countries, num_industries))
T = np.ones((num_countries, num_countries, num_industries))
pi = np.ones((num_countries, num_industries))
pi_hat = np.ones((num_countries, num_industries))
alpha = np.ones((num_countries, num_countries, num_industries))
tau_hat = np.ones((num_countries, num_countries, num_industries))
t_hat = np.ones((num_countries, num_countries, num_industries))
w_hat = np.ones(num_countries)

# Welfare function
def welfare(j, s):
    return x[j][s] + p[j]

# Government objective function
def gov_obj(params):
    tau = params.reshape((num_countries, num_countries, num_industries))
    total = 0
    for j in range(num_countries):
        for s in range(num_industries):
            total += pol_econ[j][s] * welfare(j, s)
    return -total  # We minimize, so we return the negative

# Constraint 1
def eq_12(j, s):
    total = 0
    for i in range(num_countries):
        total += (gamma[i][j][s] * (tau[i][j][s] ** (1-sigma[s]))) ** (1 / (1 - sigma[s]))
    return total

# Constraint 2 helper functions
def x2(j):
    total = 0
    for i in range(num_countries):
        for s in range(num_industries):
            total += t[i][j][s] * T[i][j][s]
    return total

def wL(j):
    term2 = 0
    for i in range(num_countries):
        for s in range(num_industries):
            term2 += t[i][j][s] * T[i][j][s]

    term3 = 0
    for s in range(num_industries):
        term3 += pi[j][s]

    return x2(j) - term2 - term3

def x2_hat(j):
    return 1  # Simplified

def complicated(j):
    total = 0
    for i in range(num_countries):
        for s in range(num_industries):
            total += (t[i][j][s] * T[i][j][s] / x2(j) * t_hat[i][j][s] * 
                      (eq_12(j, s) ** (sigma[s] - 1)) * (tau_hat[i][j][s] ** -sigma[s]) * 
                      x2_hat(j) + (pi[j][s] / x2(j) * pi_hat[j][s]))
    return total

def eq_13(j):
    term1 = wL(j) / x2(j)
    term2 = complicated(j)
    term3 = 0

    for s in range(num_industries):
        term3 += pi[j][s] / x2(j) * pi_hat[j][s]

    return term1 + term2 + term3

# Constraint 3
def eq_10(i, s):
    total = 0
    for j in range(num_countries):
        total += (alpha[i][j][s] * (tau_hat[i][j][s] ** -sigma[s]) * 
                  (w_hat[i] ** (1 - sigma[s])) * (eq_12(j, s) ** (sigma[s] - 1)) * eq_13(j))
    return total

# Constraints as a list
def constraints(params):
    tau = params.reshape((num_countries, num_countries, num_industries))
    cons = []
    for j in range(num_countries):
        for s in range(num_industries):
            cons.append(eq_12(j, s) - 1)
    for j in range(num_countries):
        cons.append(eq_13(j) - 1)
    for i in range(num_countries):
        for s in range(num_industries):
            cons.append(eq_10(i, s) - 1)
    return np.array(cons)

# Initial guess for tau
initial_tau = np.ones((num_countries, num_countries, num_industries)).flatten()

# Optimization
result = minimize(gov_obj, initial_tau, constraints={'type': 'eq', 'fun': constraints})

# Results
optimal_tau = result.x.reshape((num_countries, num_countries, num_industries))
print("Optimal tau:", optimal_tau)
# import numpy as np
# from scipy.optimize import minimize
# import pandas as pd
# import pickle

# file_path = 'T_ijs.pickle'

# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# # print(data)

# # Define countries and industries
# countries = ['China', 'Korea', 'Japan', 'USA', 'Germany']
# industries = ['gim', 'steel', 'semi', 'car']
# num_countries = len(countries)
# num_industries = len(industries)

# # Initialize dictionaries
# pol_econ = {country: {industry: 11 for industry in industries} for country in countries}
# x = {country: {industry: 34 for industry in industries} for country in countries}

# # p = {country: 0 for country in countries}
# # 2023 CPI
# P_j = { 'China': 132.2291519, 'Korea': 129.1901759, 'Japan': 111.3640359, 'USA': 139.7357936, 'Germany': 131.8924482 }

# gamma = {i: {j: {industry: 72 for industry in industries} for j in countries} for i in countries}
# tau = {i: {j: {industry: 1 for industry in industries} for j in countries} for i in countries}

# # Elasticities
# sigma = {'gim': 3.57, 'steel': 4.0, 'semi': 2.5, 'car': 1.8}
# # sigma = {industry: 1.1 for industry in industries}  # Avoid sigma[s] = 1 to prevent division by zero

# t = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}

# # T = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}
# # Trade flows data structure
# T = {
#     'China': {'Germany': {'gim': 38120.0, 'steel': 582083.0, 'semi': 1636760272.0, 'semi_memory': 119516177.0, 'car': 1833609424.0}, 
#               'Japan': {'gim': 20212670.0, 'steel': 83818503.0, 'semi': 3119372644.0, 'semi_memory': 535757854.0, 'car': 676593127.0}, 
#               'Korea': {'gim': 248805.0, 'steel': 1803944738.0, 'semi': 17758308479.0, 'semi_memory': 15219366037.0, 'car': 1212097485.0}, 
#               'USA': {'gim': 2316260.0, 'steel': 1358591.0, 'semi': 2325395161.0, 'semi_memory': 341810421.0, 'car': 2638478640.0}
#             }, 
#     'Germany': {'China': {'gim': 5240.909, 'steel': 5201023.164, 'semi': 1436904982.784, 'semi_memory': 19331662.139, 'car': 16512164176.907999}, 
#                 'Japan': {'gim': 110, 'steel': 40160.479, 'semi': 434221845.892, 'semi_memory': 693421.891, 'car': 4282380533.395}, 
#                 'Korea': {'gim': 2901.044, 'steel': 681557.131, 'semi': 628437924.905, 'semi_memory': 3842475.101, 'car': 6468871825.364}, 
#                 'USA': {'gim': 432.507, 'steel': 51998280.381, 'semi': 519539446.911, 'semi_memory': 12024057.385, 'car': 28218169963.792}
#             }, 
#     'Japan': {'China': {'gim': 1022148.922, 'steel': 777891701.735, 'semi': 6882555655.434, 'semi_memory': 4241031727.742, 'car': 6714475489.161}, 
#               'Germany': {'gim': 206532.406, 'steel': 12493288.74, 'semi': 370643557.05, 'semi_memory': 40437523.308, 'car': 2518119540.9589996}, 
#               'Korea': {'gim': 53875.234, 'steel': 1873216558.145, 'semi': 3422673575.635, 'semi_memory': 20384960.593, 'car': 746276605.623}, 
#               'USA': {'gim': 8536153.179, 'steel': 141884446.045, 'semi': 965626242.279, 'semi_memory': 57715066.618, 'car': 43064344998.029}
#             }, 
#     'USA': {'China': {'gim': 608.0, 'steel': 487349.0, 'semi': 5134039304.0, 'semi_memory': 41034340.0, 'car': 6554910660.0}, 
#             'Germany': {'gim': 861724.0, 'steel': 75461.0, 'semi': 1083274042.0, 'semi_memory': 18155007.0, 'car': 9157689622.0}, 
#             'Japan': {'gim': 439247.0, 'steel': 1199499.0, 'semi': 496029963.0, 'semi_memory': 14758843.0, 'car': 1277318495.0}, 
#             'Korea': {'gim': 169675.0, 'steel': 180376.0, 'semi': 2204298430.0, 'semi_memory': 15309944.0, 'car': 2740915359.0}
#             }, 
#     'Korea': {'China': {'gim': 48062453.0, 'steel': 543488181.0, 'semi': 65634823868.0, 'semi_memory': 34451077419.0, 'car': 313316807.0}, 
#               'Germany': {'gim': 1055190.583, 'steel': 52080498.965, 'semi': 1280019502.842, 'semi_memory': 156173609.446, 'car': 11664726.954}, 
#               'Japan': {'gim': 104775627.791, 'steel': 921335767.82, 'semi': 1572825138.534, 'semi_memory': 419536125.294, 'car': 495404.448}, 
#               'USA': {'gim': 33963781.0, 'steel': 581315367.0, 'semi': 2400493160.0, 'semi_memory': 236493837.0, 'car': 883492635.0}
#             }
#     }

# pi = {country: {industry: 9 for industry in industries} for country in countries}
# pi_hat = {country: {industry: 3 for industry in industries} for country in countries}
# alpha = {i: {j: {industry: 6 for industry in industries} for j in countries} for i in countries}
# tau_hat = {i: {j: {industry: 1 for industry in industries} for j in countries} for i in countries}
# t_hat = {i: {j: {industry: 1 for industry in industries} for j in countries} for i in countries}
# w_hat = {country: 1 for country in countries}

# # Example values (you can fill with actual data or random values)
# for country in countries:
#     for industry in industries:
#         pol_econ[country][industry] = np.random.rand()
#         x[country][industry] = np.random.rand()
#     P_j[country] = np.random.rand()

# for i in countries:
#     for j in countries:
#         for industry in industries:
#             gamma[i][j][industry] = np.random.rand()
#             tau[i][j][industry] = np.random.rand() + 1
#             t[i][j][industry] = np.random.rand()
#             # T[i][j][industry] = np.random.rand()
#             alpha[i][j][industry] = np.random.rand()
#             tau_hat[i][j][industry] = np.random.rand() + 1
#             t_hat[i][j][industry] = np.random.rand() + 1

# for country in countries:
#     for industry in industries:
#         pi[country][industry] = np.random.rand()
#         pi_hat[country][industry] = np.random.rand()

# # Welfare function
# def welfare(j, s):
#     return x[j][s] + P_j[j]

# # Government objective function for country j
# def gov_obj(tau_js, j):
#     tau_copy = tau.copy()
#     for i, industry in enumerate(industries):
#         for k, country in enumerate(countries):
#             if j != country:
#                 tau_copy[country][j][industry] = tau_js[i * num_countries + k]
#     total = 0
#     for s in industries:
#         total += pol_econ[j][s] * welfare(j, s)
#     return -total  # We minimize, so we return the negative

# # Constraint 1 for country j and industry s
# def eq_12(j, s):
#     total = 0
#     for i in countries:
#         if i != j:
#             total += (gamma[i][j][s] * (tau[i][j][s] ** (1 - sigma[s]))) ** (1 / (1 - sigma[s]))
#     return total

# # Constraint 2 helper functions
# def x2(j):
#     total = 0
#     for i in countries:
#         for s in industries:
#             if i != j:
#                 total += t[i][j][s] * T[i][j][s]
#     return total

# def wL(j):
#     term2 = 0
#     for i in countries:
#         for s in industries:
#             if i != j:
#                 term2 += t[i][j][s] * T[i][j][s]

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
#             if i != j:
#                 total += (t[i][j][s] * T[i][j][s] / x2(j) * t_hat[i][j][s] * 
#                         (eq_12(j, s) ** (sigma[s] - 1)) * (tau_hat[i][j][s] ** -sigma[s]) * 
#                         x2_hat(j) + (pi[j][s] / x2(j) * pi_hat[j][s]))
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
#         if i != j:
#             total += (alpha[i][j][s] * (tau_hat[i][j][s] ** -sigma[s]) * 
#                     (w_hat[i] ** (1 - sigma[s])) * (eq_12(j, s) ** (sigma[s] - 1)) * eq_13(j))
#     return total

# # Constraints as a list for country j
# def constraints(tau_js, j):
#     tau_copy = tau.copy()
#     for i, industry in enumerate(industries):
#         for k, country in enumerate(countries):
#             if country != j:
#                 tau_copy[country][j][industry] = tau_js[i * num_countries + k]
#     cons = []
#     for s in industries:
#         cons.append({'type': 'eq', 'fun': lambda tau_js, j=j, s=s: eq_12(j, s) - 1})
#     cons.append({'type': 'eq', 'fun': lambda tau_js, j=j: eq_13(j) - 1})
#     for i in countries:
#         for s in industries:
#             cons.append({'type': 'eq', 'fun': lambda tau_js, i=i, s=s: eq_10(i, s) - 1})
#     return cons

# # Optimize tariffs for each country i (home country)
# for i in countries:
#     optimal_taus = {j: {industry: 0 for industry in industries} for j in countries if j != i}
#     for j in countries:
#         if j == i:
#             continue
#         initial_tau_js = np.ones((num_countries, num_industries)).flatten()
#         result = minimize(gov_obj, initial_tau_js, args=(j,), constraints=constraints(initial_tau_js, j))
#         for k, industry in enumerate(industries):
#             for m, country in enumerate(countries):
#                 if country != i:
#                     optimal_taus[country][industry] = result.x[k * num_countries + m]
    
#     print(f"Optimal tariffs for {i} as the home country:")
#     print(pd.DataFrame(optimal_taus))
#     print("\n")


# to check if actual tariffs get calculated
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import pickle

file_path = 'T_ijs.pickle'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Define countries and industries
countries = ['China', 'Korea', 'Japan', 'USA', 'Germany']
industries = ['gim', 'steel', 'semi', 'car']
num_countries = len(countries)
num_industries = len(industries)

# Initialize dictionaries with varied values
pol_econ = {country: {industry: np.random.rand() for industry in industries} for country in countries}
x = {country: {industry: np.random.rand() for industry in industries} for country in countries}
P_j = {country: np.random.rand() for country in countries}
gamma = {i: {j: {industry: np.random.rand() for industry in industries} for j in countries} for i in countries}
tau = {i: {j: {industry: np.random.rand() + 1 for industry in industries} for j in countries} for i in countries}
sigma = {'gim': 3.57, 'steel': 4.0, 'semi': 2.5, 'car': 1.8}
t = {i: {j: {industry: np.random.rand() for industry in industries} for j in countries} for i in countries}
# T = data  # Assuming data is loaded from the pickle file correctly
pi = {country: {industry: np.random.rand() for industry in industries} for country in countries}
pi_hat = {country: {industry: np.random.rand() for industry in industries} for country in countries}
alpha = {i: {j: {industry: np.random.rand() for industry in industries} for j in countries} for i in countries}
tau_hat = {i: {j: {industry: np.random.rand() + 1 for industry in industries} for j in countries} for i in countries}
t_hat = {i: {j: {industry: np.random.rand() + 1 for industry in industries} for j in countries} for i in countries}
w_hat = {country: 1 for country in countries}

T = {
    'China': {'Germany': {'gim': 38120.0, 'steel': 582083.0, 'semi': 1636760272.0, 'semi_memory': 119516177.0, 'car': 1833609424.0}, 
              'Japan': {'gim': 20212670.0, 'steel': 83818503.0, 'semi': 3119372644.0, 'semi_memory': 535757854.0, 'car': 676593127.0}, 
              'Korea': {'gim': 248805.0, 'steel': 1803944738.0, 'semi': 17758308479.0, 'semi_memory': 15219366037.0, 'car': 1212097485.0}, 
              'USA': {'gim': 2316260.0, 'steel': 1358591.0, 'semi': 2325395161.0, 'semi_memory': 341810421.0, 'car': 2638478640.0}
            }, 
    'Germany': {'China': {'gim': 5240.909, 'steel': 5201023.164, 'semi': 1436904982.784, 'semi_memory': 19331662.139, 'car': 16512164176.907999}, 
                'Japan': {'gim': 110, 'steel': 40160.479, 'semi': 434221845.892, 'semi_memory': 693421.891, 'car': 4282380533.395}, 
                'Korea': {'gim': 2901.044, 'steel': 681557.131, 'semi': 628437924.905, 'semi_memory': 3842475.101, 'car': 6468871825.364}, 
                'USA': {'gim': 432.507, 'steel': 51998280.381, 'semi': 519539446.911, 'semi_memory': 12024057.385, 'car': 28218169963.792}
            }, 
    'Japan': {'China': {'gim': 1022148.922, 'steel': 777891701.735, 'semi': 6882555655.434, 'semi_memory': 4241031727.742, 'car': 6714475489.161}, 
              'Germany': {'gim': 206532.406, 'steel': 12493288.74, 'semi': 370643557.05, 'semi_memory': 40437523.308, 'car': 2518119540.9589996}, 
              'Korea': {'gim': 53875.234, 'steel': 1873216558.145, 'semi': 3422673575.635, 'semi_memory': 20384960.593, 'car': 746276605.623}, 
              'USA': {'gim': 8536153.179, 'steel': 141884446.045, 'semi': 965626242.279, 'semi_memory': 57715066.618, 'car': 43064344998.029}
            }, 
    'USA': {'China': {'gim': 608.0, 'steel': 487349.0, 'semi': 5134039304.0, 'semi_memory': 41034340.0, 'car': 6554910660.0}, 
            'Germany': {'gim': 861724.0, 'steel': 75461.0, 'semi': 1083274042.0, 'semi_memory': 18155007.0, 'car': 9157689622.0}, 
            'Japan': {'gim': 439247.0, 'steel': 1199499.0, 'semi': 496029963.0, 'semi_memory': 14758843.0, 'car': 1277318495.0}, 
            'Korea': {'gim': 169675.0, 'steel': 180376.0, 'semi': 2204298430.0, 'semi_memory': 15309944.0, 'car': 2740915359.0}
            }, 
    'Korea': {'China': {'gim': 48062453.0, 'steel': 543488181.0, 'semi': 65634823868.0, 'semi_memory': 34451077419.0, 'car': 313316807.0}, 
              'Germany': {'gim': 1055190.583, 'steel': 52080498.965, 'semi': 1280019502.842, 'semi_memory': 156173609.446, 'car': 11664726.954}, 
              'Japan': {'gim': 104775627.791, 'steel': 921335767.82, 'semi': 1572825138.534, 'semi_memory': 419536125.294, 'car': 495404.448}, 
              'USA': {'gim': 33963781.0, 'steel': 581315367.0, 'semi': 2400493160.0, 'semi_memory': 236493837.0, 'car': 883492635.0}
            }
    }

# Welfare function
def welfare(j, s):
    return x[j][s] + P_j[j]

# Government objective function for country j
def gov_obj(tau_js, j):
    tau_copy = {i: {k: {industry: tau[i][k][industry] for industry in industries} for k in countries} for i in countries}
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
    return total - 1  # Constraint to be equal to 1

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

    return term1 + term2 + term3 - 1  # Constraint to be equal to 1

# Constraint 3 for country i and industry s
def eq_10(i, s):
    total = 0
    for j in countries:
        if i != j:
            total += (alpha[i][j][s] * (tau_hat[i][j][s] ** -sigma[s]) * 
                    (w_hat[i] ** (1 - sigma[s])) * (eq_12(j, s) ** (sigma[s] - 1)) * eq_13(j))
    return total - 1  # Constraint to be equal to 1

# Constraints as a list for country j
def constraints(tau_js, j):
    tau_copy = {i: {k: {industry: tau[i][k][industry] for industry in industries} for k in countries} for i in countries}
    for i, industry in enumerate(industries):
        for k, country in enumerate(countries):
            if country != j:
                tau_copy[country][j][industry] = tau_js[i * num_countries + k]
    cons = []
    for s in industries:
        cons.append({'type': 'eq', 'fun': lambda tau_js, j=j, s=s: eq_12(j, s)})
    cons.append({'type': 'eq', 'fun': lambda tau_js, j=j: eq_13(j)})
    for i in countries:
        for s in industries:
            cons.append({'type': 'eq', 'fun': lambda tau_js, i=i, s=s: eq_10(i, s)})
    return cons

# Optimize tariffs for each country i (home country)
for i in countries:
    optimal_taus = {j: {industry: 0 for industry in industries} for j in countries if j != i}
    for j in countries:
        if j == i:
            continue
        initial_tau_js = np.random.rand(num_countries * num_industries) * 0.5 + 1.0
        result = minimize(gov_obj, initial_tau_js, args=(j,), constraints=constraints(initial_tau_js, j))
        for k, industry in enumerate(industries):
            for m, country in enumerate(countries):
                if country != i:
                    optimal_taus[country][industry] = result.x[k * num_countries + m]
    
    print(f"Optimal tariffs for {i} as the home country:")
    print(pd.DataFrame(optimal_taus))
    print("\n")

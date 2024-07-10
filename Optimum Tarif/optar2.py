import numpy as np
from scipy.optimize import minimize

# Define countries and industries
countries = ['China', 'Korea', 'Japan', 'US', 'Germany']
industries = ['GIM', 'Steel', 'Semiconductor', 'Car']
num_countries = len(countries)
num_industries = len(industries)

# Initialize 3D dictionaries
X_is = {(i, s): np.random.rand() for i in range(num_countries) for s in range(num_industries)}  # Export matrix
P_i = {i: np.random.rand() for i in range(num_countries)}  # Price index for each country
T_ijs = {(i, j, s): np.random.rand() for i in range(num_countries) for j in range(num_countries) for s in range(num_industries)}  # Initial tariff matrix
X_ijs = {(i, j, s): np.random.rand() for i in range(num_countries) for j in range(num_countries) for s in range(num_industries)}  # Trade flow matrix

# Equation 10: Calculate share of industry s in country i's total exports
def calculate_pi_is(X_is):
    X_i = {i: sum(X_is[i, s] for s in range(num_industries)) for i in range(num_countries)}
    pi_is = {(i, s): X_is[i, s] / X_i[i] for i in range(num_countries) for s in range(num_industries)}
    return pi_is

pi_is = calculate_pi_is(X_is)

# Equation 11: Calculate price index P_j
def calculate_P_j(pi_is, P_i):
    P_j = {j: sum(pi_is[i, s] * P_i[i] for i in range(num_countries)) for j in range(num_countries) for s in range(num_industries)}
    return P_j

P_j = calculate_P_j(pi_is, P_i)

# Equation 12: Calculate tariff rate tau_ij
def calculate_tau_ij(T_ijs, P_j):
    tau_ij = {(i, j, s): T_ijs[i, j, s] / P_j[j] for i in range(num_countries) for j in range(num_countries) for s in range(num_industries)}
    return tau_ij

tau_ij = calculate_tau_ij(T_ijs, P_j)

# Equation 13: Check equilibrium condition
def check_equilibrium(tau_ij, X_ijs, T_ijs):
    lhs = {j: sum(tau_ij[i, j, s] * X_ijs[i, j, s] for i in range(num_countries) for s in range(num_industries)) for j in range(num_countries)}
    rhs = {i: sum(T_ijs[i, j, s] * X_ijs[i, j, s] for j in range(num_countries) for s in range(num_industries)) for i in range(num_countries)}
    return np.allclose(list(lhs.values()), list(rhs.values()))

# Objective function to maximize total tariff revenue
def objective(tau_ij_flat):
    tau_ij = {(i, j, s): tau_ij_flat[i * num_countries * num_industries + j * num_industries + s] for i in range(num_countries) for j in range(num_countries) for s in range(num_industries)}
    revenue = sum(tau_ij[i, j, s] * X_ijs[i, j, s] for i in range(num_countries) for j in range(num_countries) for s in range(num_industries))
    return -revenue  # Negative because we minimize in scipy.optimize

# Constraints: Ensure equilibrium condition holds
def equilibrium_constraint(tau_ij_flat):
    tau_ij = {(i, j, s): tau_ij_flat[i * num_countries * num_industries + j * num_industries + s] for i in range(num_countries) for j in range(num_countries) for s in range(num_industries)}
    lhs = {j: sum(tau_ij[i, j, s] * X_ijs[i, j, s] for i in range(num_countries) for s in range(num_industries)) for j in range(num_countries)}
    rhs = {i: sum(T_ijs[i, j, s] * X_ijs[i, j, s] for j in range(num_countries) for s in range(num_industries)) for i in range(num_countries)}
    return list(lhs.values()) - list(rhs.values())

# Initial guess for tau_ij
tau_ij_initial = [T_ijs[i, j, s] for i in range(num_countries) for j in range(num_countries) for s in range(num_industries)]

# Constraints dictionary for scipy.optimize
constraints = [{'type': 'eq', 'fun': equilibrium_constraint}]

# Optimize
result = minimize(objective, tau_ij_initial, constraints=constraints, method='SLSQP')

# Reshape result to original dictionary form
optimal_tau_ij = {(i, j, s): result.x[i * num_countries * num_industries + j * num_industries + s] for i in range(num_countries) for j in range(num_countries) for s in range(num_industries)}

# Output results
print("Optimal tariff rates (tau_ij):")
for (i, j, s), value in optimal_tau_ij.items():
    print(f"tau[{countries[i]}, {countries[j]}, {industries[s]}] = {value:.4f}")

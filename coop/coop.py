import numpy as np
from scipy.optimize import minimize
import pandas as pd

# (Previous code for setting up countries, industries, and data structures remains the same)

# Nash equilibrium tariffs (assuming we have calculated these already)
nash_tariffs = {i: {j: {industry: 1.0 for industry in industries} for j in countries if j != i} for i in countries}

# Function to calculate welfare for a country
def calculate_welfare(country, tariffs):
    # Implement welfare calculation based on your model
    # This is a placeholder - you'll need to replace this with your actual welfare calculation
    return sum(sum(tariffs[country][j].values()) for j in countries if j != country)

# Function to calculate welfare gains from Nash to Cooperative
def welfare_gains(cooperative_tariffs):
    gains = {}
    for country in countries:
        nash_welfare = calculate_welfare(country, nash_tariffs)
        coop_welfare = calculate_welfare(country, cooperative_tariffs)
        gains[country] = coop_welfare - nash_welfare
    return gains

# Objective function for cooperative tariffs
def cooperative_objective(tau_all):
    cooperative_tariffs = process_tariffs(tau_all)
    gains = welfare_gains(cooperative_tariffs)
    
    # We want to maximize the minimum gain to ensure even distribution
    return -min(gains.values())

# Helper function to process tariffs
def process_tariffs(tau_all):
    tariffs = {i: {j: {industry: 0 for industry in industries} for j in countries if j != i} for i in countries}
    idx = 0
    for i in countries:
        for j in countries:
            if i != j:
                for industry in industries:
                    tariffs[i][j][industry] = tau_all[idx]
                    idx += 1
    return tariffs

# Constraints function (reuse from previous code)
def cooperative_constraints(tau_all):
    # (Same as before)
    pass

# Initial guess (you might want to start from Nash equilibrium tariffs)
initial_cooperative_tariffs = [nash_tariffs[i][j][industry] 
                               for i in countries 
                               for j in countries if j != i 
                               for industry in industries]

# Optimize cooperative tariffs
result = minimize(cooperative_objective, 
                  initial_cooperative_tariffs, 
                  constraints=cooperative_constraints(initial_cooperative_tariffs),
                  method='SLSQP',  # You might need to experiment with different methods
                  options={'maxiter': 1000})

# Process results
cooperative_tariffs = process_tariffs(result.x)

# Print cooperative tariffs
for i in countries:
    print(f"Cooperative tariffs for {i}:")
    print(pd.DataFrame(cooperative_tariffs[i]))
    print("\n")

# Calculate and print welfare gains
gains = welfare_gains(cooperative_tariffs)
for country, gain in gains.items():
    print(f"Welfare gain for {country}: {gain}")

# Check if gains are evenly distributed
min_gain = min(gains.values())
max_gain = max(gains.values())
if max_gain - min_gain < 1e-6:  # Some small threshold for floating-point comparison
    print("Welfare gains are evenly distributed across countries.")
else:
    print("Warning: Welfare gains are not evenly distributed.")
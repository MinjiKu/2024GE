import numpy as np
import pandas as pd

industries = ['steel', 'semi', 'car']
countries = ['China', 'Korea', 'Japan', 'USA', 'Germany']

tau = {
    'China': {
        'Korea': {'gim': 1.18, 'steel': 1, 'semi': 1, 'car': 1.059},
        'Japan': {'gim': 1.4, 'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'gim': 1, 'steel': 1, 'semi': 1.25, 'car': 1.2275},
        'Germany': {'gim': 1, 'steel': 1.359, 'semi': 1, 'car': 1.03}
    },
    'Korea': {
        'China': {'gim': 1.08, 'steel': 1.005, 'semi': 1.01, 'car': 1.04},
        'Japan': {'gim': 1.4, 'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'gim': 1, 'steel': 1, 'semi': 1, 'car': 1},
        'Germany': {'gim': 1, 'steel': 1, 'semi': 1, 'car': 1}
    },
    'Japan': {
        'China': {'gim': 1.175, 'steel': 1.044, 'semi': 1, 'car': 1.077},
        'Korea': {'gim': 1.2, 'steel': 1, 'semi': 1, 'car': 1.065},
        'USA': {'gim': 1, 'steel': 1, 'semi': 1, 'car': 1.0212},
        'Germany': {'gim': 1, 'steel': 1, 'semi': 1, 'car': 1}
    },
    'USA': {
        'China': {'gim': 1.2, 'steel': 1.05, 'semi': 1, 'car': 1.085},
        'Korea': {'gim': 1, 'steel': 1, 'semi': 1, 'car': 1},
        'Japan': {'gim': 1.4, 'steel': 1, 'semi': 1, 'car': 1},
        'Germany': {'gim': 1, 'steel': 1, 'semi': 1, 'car': 1.03}
    },
    'Germany': {
        'China': {'gim': 1.2, 'steel': 1.05, 'semi': 1, 'car': 1.085},
        'Korea': {'gim': 1, 'steel': 1, 'semi': 1, 'car': 1},
        'Japan': {'gim': 1.4, 'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'gim': 1, 'steel': 1, 'semi': 1, 'car': 1.0212}
    }
}

# # Initialize the 2D array for tau_js
# tau_js = np.zeros((len(industries), len(countries)))

# # Populate the 2D array with aggregated values from tau
# for i, industry in enumerate(industries):
#     for j, country in enumerate(countries):
#         # Average the values for the given industry and country across all source countries
#         values = [tau[source_country][country][industry] for source_country in countries if country in tau[source_country]]
#         tau_js[i, j] = np.mean(values)

# # Display the 2D array
# print("tau_js:")
# print(tau_js)
# Initialize the 2D array for tau_js
tau_js = np.zeros((len(industries), len(countries)))

# Populate the 2D array with aggregated values from tau
for i, industry in enumerate(industries):
    for j, country in enumerate(countries):
        # Average the values for the given industry and country across all source countries
        values = [tau[source_country][country][industry] for source_country in countries if country in tau[source_country]]
        tau_js[i, j] = np.mean(values)

# Create a DataFrame for better readability
tau_js_df = pd.DataFrame(tau_js, index=industries, columns=countries)

# Print the DataFrame
print("tau_js:")
print(tau_js_df)
import matplotlib.pyplot as plt

# Define the iteration numbers (including the first iteration)
iterations = [1, 2, 3, 4, 5]

# Starting welfare values for each iteration (including the first iteration)
starting_welfare = [
    {'China': 996847543.7464023, 'Korea': 1285179394.0915222, 'Japan': 1416340729.3391767, 'USA': 888335325.7477989, 'Germany': 1373532787.7835402},
    {'China': 957854732.2080762, 'Korea': 1255528518.9338794, 'Japan': 1414800300.5951066, 'USA': 943034776.0884926, 'Germany': 1383012689.8339343},
    {'China': 957854732.2068509, 'Korea': 1255528518.9372416, 'Japan': 1414800300.5993588, 'USA': 943034776.0867236, 'Germany': 1383012689.8373446},
    {'China': 957854732.2072086, 'Korea': 1255528518.9375994, 'Japan': 1414800300.5997167, 'USA': 943034776.0867236, 'Germany': 1383012689.8377023},
    {'China': 957854732.20723, 'Korea': 1255528518.9376206, 'Japan': 1414800300.599738, 'USA': 943034776.0871027, 'Germany': 1383012689.8373446}
]

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(14, 7))

# Define markers and line styles for each country to differentiate them
markers = {
    'China': 'o',
    'Korea': 's',
    'Japan': '^',
    'USA': 'D',
    'Germany': 'x'
}

line_styles = {
    'China': '-',
    'Korea': '--',
    'Japan': '-.',
    'USA': ':',
    'Germany': '-'
}

# Plot starting welfare for each country
for country in starting_welfare[0].keys():
    welfare_values = [starting_welfare[i][country] for i in range(len(starting_welfare))]
    ax.plot(iterations, welfare_values, label=country, marker=markers[country], linestyle=line_styles[country], linewidth=2)

# Set labels and title
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Welfare Value', fontsize=12)
ax.set_title('Welfare for Each Country Over Iterations', fontsize=14)

# Add legend with increased font size for better readability
ax.legend(fontsize=12)

# Add grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Tight layout for better spacing
plt.tight_layout()

# Save the plot to a file
fig.savefig('welfare_all_countries.png', dpi=300)

# Display the plot (optional)
plt.show()

# Close the figure
plt.close(fig)

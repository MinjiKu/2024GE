# -------------------------- Final Code --------------------------------

import sys
import os
import torch
import torch.optim as optim

# Get the absolute path of the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(script_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)
import var

# def flatten_dict(tau_dict):
#     flat_list = torch.tensor([], requires_grad=True)
#     flat_list.requires_grad(True)

#     for i in tau_dict:
#         for industry in tau_dict[i][j]:  # Ensure this accesses correct indices
#             x = torch.tensor(tau_dict[i][j][industry], requires_grad=True)
#             x.requires_grad(True)
#             flat_list.append(x)
#     return torch.tensor(flat_list, dtype=torch.float32)

def flatten_dict(tau_dict):
    flat_list = []
    for i in tau_dict:
        for industry in tau_dict[i][j]:
            flat_list.append(tau_dict[i][j][industry])
    return torch.tensor(flat_list, dtype=torch.float32)

def unflatten_dict(flat_tensor, j):
    unflattened_dict = {}
    index = 0
    for i in var.countries:
        if i != j:
            unflattened_dict[i] = {j: {}}
            for industry in var.industries:
                unflattened_dict[i][j][industry] = flat_tensor[index].item()
                index += 1
    return unflattened_dict

def update_economic_variables(tau, j):
    # Convert tau to dictionary form if it's a tensor
    if isinstance(tau, torch.Tensor):
        tau = unflatten_dict(tau, j)

    for i in tau.keys():
        if i == j: continue
        for industry in var.industries:
            var.t[i][j][industry] = max(tau[i][j][industry] - 1, 1e-10)

    var.fill_gamma()
    var.fill_alpha()
    var.fill_pi()

    for i in var.countries:
        for j in var.countries:
            if i != j:
                for industry in var.industries:
                    previous_tau = var.factual_tau[i][j][industry]
                    change_rate = (var.tau[i][j][industry] - previous_tau) / previous_tau if previous_tau != 0 else 0
                    var.p_ijs[i][j][industry] *= (1 + change_rate)
                    var.T[i][j][industry] *= (1 + change_rate * var.de[industry])

def calc_x(j, s):
    TR = 0
    for i in var.countries:
        if i == j: continue
        TR += var.t[i][j][s] * var.T[i][j][s]
    
    return var.w[j] * var.L_js[j][s] + var.pi[j][s] + var.L_js[j][s]/var.L_j[j] * TR

# Welfare function
def calc_welfare(j, s):
    return calc_x(j, s) / var.P_j[j]

def gov_obj(tau, j):
    update_economic_variables(tau, j)
    total = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)  # Initialize total as a tensor
    for s in var.industries:
        total += var.pol_econ[j][s] * calc_welfare(j, s)
    return -total  # Return a tensor that requires gradients

welfare_history = {i: {j: {industry: [] for industry in var.industries} for j in var.countries if j != i} for i in var.countries}

def optimize_for_importer(j, var):
    # Flatten the tau structure for the current importer
    initial_tau_dict = {
        i: {j: {s: var.tau[i][j][s] for s in var.industries}}
        for i in var.countries if i != j
    }
    
    initial_tau = flatten_dict(initial_tau_dict)
    
    # Convert initial_tau to a torch tensor
    tau_tensor = initial_tau.clone().detach().requires_grad_(True)
    
    # Define the optimizer (Adam is used here, but you can choose others)
    optimizer = optim.Adam([tau_tensor], lr=0.01)

    # Perform optimization
    for step in range(100):  # Example: 100 steps of optimization
        optimizer.zero_grad()
        loss = torch.tensor(gov_obj(tau_tensor, j), requires_grad=True)  # Calculate the loss (negative objective)
        loss.requires_grad_(True)
        
        loss.backward()  # Backpropagate the gradients
        optimizer.step()  # Update the tau_tensor

        # Print the tau_tensor values and gradients for debugging
        print(f"Iteration {step + 1} for country {j}:")
        print(f"tau_tensor: {tau_tensor}")
        print(f"Gradients: {tau_tensor.grad}")
    
    # After optimization, map the results back to the original structure
    optimized_tau = unflatten_dict(tau_tensor, j)
    for i in var.countries:
        if i != j:
            for s in var.industries:
                var.tau[i][j][s] = optimized_tau[i][j][s]
                welfare_history[i][j][s].append(-loss.item())  # Append the welfare value

    # print("tau_tensor")
    # print(tau_tensor)
    # print("welfare_history")
    # print(welfare_history)

iteration = 10

# Usage in the main loop:
for iter in range(iteration):
    for j in var.countries:
        print(f"Optimizing tariffs for imports into country: {j}")
        optimize_for_importer(j, var)
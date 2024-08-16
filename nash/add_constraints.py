import sys
import os


# Get the absolute path of the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(script_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)
import var
# # to check if actual tariffs get calculated
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle
np.set_printoptions(precision=15, suppress=False)
# Times New Roman 폰트를 사용하도록 설정
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] + rcParams['font.serif']

var.fill_gamma()
var.fill_pi()
factual_pi = var.pi.copy() #factual var.pi 보존

def get_flat_index(i, j, s):
    countries = ['China', 'Korea', 'Japan', 'USA', 'Germany']
    industries = ['steel', 'semi', 'car']

    num_industries = len(industries)
    
    i_index = countries.index(i)
    j_index = countries.index(j)
    
    # i와 j가 동일한 리스트를 사용하고 있으므로,
    # i의 위치에 따라 j의 인덱스를 조정해야 함
    if j_index > i_index:
        j_index -= 1  # i보다 뒤에 있으면, 인덱스에서 1을 빼야 맞음

    s_index = industries.index(s)

    # 평탄화된 리스트에서의 최종 인덱스 계산
    flat_index = i_index * (len(countries) - 1) * num_industries + j_index * num_industries + s_index

    return flat_index
# var.fill_alpha()
def update_economic_variables(tau, previous_tau):
    for i in var.countries:
        for j in var.countries:
            if i == j: continue
            for industry in var.industries:
                #print("print here", tau)
                var.t[i][j][industry] = max(tau[i][j][industry] - 1.0, 1e-10)

    # Update p_ijs and T based on the new values of t
    for i in var.countries:
        for j in var.countries:
            if i != j:
                for industry in var.industries:
                    change_rate = (var.tau[i][j][industry] - previous_tau[i][j][industry]) / previous_tau[i][j][industry] if previous_tau[i][j][industry] != 0 else 0
                    var.p_ijs[i][j][industry] *= (1 + change_rate)
                    var.T[i][j][industry] *= (1 - change_rate * var.de[industry])
    
    # Recalculate gamma, pi, and alpha based on the updated t values
    var.fill_gamma()
    var.fill_alpha()
    var.fill_pi()

def calc_x(j, s):
    sum = 0
    TR = 0
    for i in var.countries:
        if i == j: continue
        TR += var.t[i][j][s] * var.T[i][j][s]
    
    sum = var.w[j] * var.L_js[j][s] + var.pi[j][s] + var.L_js[j][s]/var.L_j[j] * TR
    
    return sum

# Welfare function
def calc_welfare(j, s):
    return calc_x(j, s) / var.P_j[j]

def gov_obj(tau, j):
    # Update economic variables based on the current tau_js
    #update_economic_variables(tau, j)

    total = 0
    for s in var.industries:
        total += var.pol_econ[j][s] * calc_welfare(j, s)

    return -total  # We minimize, so we return the negative

def flatten_dict(tau_dict):
    flat_list = []
    for i in tau_dict.keys():  # Iterate over all exporter keys
        for j in tau_dict[i].keys():  # Iterate over all importer keys
            for industry in tau_dict[i][j].keys():  # Iterate over all industries
                flat_list.append(tau_dict[i][j][industry])
    return flat_list

def unflatten_dict(flat_list):
    unflattened_dict = {}
    index = 0
    
    for i in var.countries:  # Iterate over all exporter keys
        unflattened_dict[i] = {}
        for j in var.countries:  # Iterate over all importer keys
            if i != j:
                unflattened_dict[i][j] = {}
                for industry in var.industries:  # Iterate over all industries
                    unflattened_dict[i][j][industry] = flat_list[index]
                    index += 1
    
    return unflattened_dict

def eq_10(i, s, unflat_tau):
    total = 0
    for j in var.countries:
        if i != j:
            flat_index = get_flat_index(i, j, s)
            tau_value = max(unflat_tau[flat_index],1)
            total += (var.alpha[i][j][s] * (tau_value ** -var.sigma[s]) * 
                      (var.w[i] ** (1 - var.sigma[s])) * (var.P_hat[j][s] ** (var.sigma[s] - 1)) * var.X_hat[j])
    
    return total - var.pi_hat[i][s]

def eq_12(j, s, unflat_tau):
    total = 0
    for i in var.countries:
        if i != j:
            flat_index = get_flat_index(i, j, s)
            tau_value = max(unflat_tau[flat_index], 1)
            total += (var.gamma[i][j][s] * (tau_value ** (1 - var.sigma[s]))) ** (1 / (1 - var.sigma[s]))
    var.P_hat[j][s] = total
    return var.P_hat[j][s] - total

def x2(j, unflat_tau):
    total = 0
    for i in var.countries:
        for s in var.industries:
            if i != j:
                flat_index = get_flat_index(i, j, s)
                total += unflat_tau[flat_index] * var.T[i][j][s]
    return total

def wL(j, unflat_tau):
    term2 = 0
    for i in var.countries:
        for s in var.industries:
            if i != j:
                flat_index = get_flat_index(i, j, s)
                term2 += unflat_tau[flat_index] * var.T[i][j][s]

    term3 = 0
    for s in var.industries:
        term3 += var.pi[j][s]

    return x2(j, unflat_tau) - term2 - term3

def term3(j, unflat_tau):
    total = 0
    for s in var.industries:
        for i in var.countries:
            if i != j:
                if var.tau_hat[i][j][s]<1e-10:
                    var.tau_hat[i][j][s]=1e-10
                total += (var.pi[j][s] / x2(j, unflat_tau) * var.pi_hat[j][s]) * var.alpha[j][i][s] * (abs(var.tau_hat[j][i][s]) ** -var.sigma[s]) * (var.w[i] ** (1 - var.sigma[s])) * (var.P_hat[j][s] ** (var.sigma[s] - 1))
    return total

def complicated(j, unflat_tau):
    total = 0
    for i in var.countries:
        for s in var.industries:
            if i != j:
                flat_index = get_flat_index(i, j, s)
                total += (unflat_tau[flat_index] * var.T[i][j][s] / x2(j, unflat_tau) * var.t_hat[i][j][s] * 
                        (var.P_hat[j][s] ** (var.sigma[s] - 1)) * (abs(var.tau_hat[i][j][s]) ** -var.sigma[s]) + 
                        (var.pi[j][s] / x2(j, unflat_tau) * var.pi_hat[j][s]))
    return total

def eq_13(j, unflat_tau):
    epsilon = 1e-10
    term1 = wL(j, unflat_tau) / (x2(j, unflat_tau) + epsilon)
    term2 = complicated(j, unflat_tau)
    term3_val = term3(j, unflat_tau)

    aggregated_x = 0
    x_j = {j:1 for j in var.countries}
    for s in var.industries:
        aggregated_x += calc_x(j, s)
        x_j[j] += var.x[j][s]

    var.X_hat[j] = term1 + term2 + term3_val
    return term1 + term2 + term3_val - aggregated_x / x_j[j]


def update_hats(tau, t, pi): #갱신된 값이 인자로 들어감
    # global pi_hat, tau_hat, t_hat, factual_tau, factual_pi, factual_t
    global factual_pi
    for i in var.countries:
        for j in var.countries:
            if i != j:
                for s in var.industries:
                    var.tau_hat[i][j][s] = abs(tau[i][j][s] / var.factual_tau[i][j][s])
                    var.t_hat[i][j][s] = abs(t[i][j][s] / var.factual_t[i][j][s])
    for j in var.countries:
        for s in var.industries:
            var.pi_hat[j][s] = abs(pi[j][s] / factual_pi[j][s])


def constraints(unflat_tau, j):
    cons = []
    
    # Constraint 1: eq_12
    for s in var.industries:
        cons.append({'type': 'eq', 'fun': lambda unflat_tau, j=j, s=s: eq_12(j, s, unflat_tau)})
    
    # Constraint 2: eq_13
    cons.append({'type': 'eq', 'fun': lambda unflat_tau, j=j: eq_13(j, unflat_tau)})
    
    # # Constraint 3: eq_10 for each country i and industry s
    # for i in var.countries:
    #     if i != j:
    #         for s in var.industries:
    #             cons.append({'type': 'eq', 'fun': lambda unflat_tau, i=i, s=s: eq_10(i, s, unflat_tau)})
    
    return cons

def calculate_average_change(current_tau, previous_tau):
    total_change = 0
    count = 0
    for i in var.countries:
        for j in var.countries:
            if i != j:
                for s in var.industries:
                    previous_value = previous_tau[i][j][s]
                    current_value = current_tau[i][j][s]
                    if previous_value > 0:  # 0으로 나누지 않도록
                        change = abs((current_value - previous_value) / previous_value)
                        total_change += change
                        count += 1
    return total_change / count if count > 0 else 0


# Ensure the directory exists
output_dir = "visualization_img"
os.makedirs(output_dir, exist_ok=True)

def optimize_for_importer(j):
    # Flatten the tau structure for the current importer
    initial_tau = flatten_dict({
        i: {j: {s: var.tau[i][j][s] for s in var.industries} for j in var.countries if i != j}
        for i in var.countries
    })

    def importer_gov_obj(flat_tau):
        unflattened_tau = unflatten_dict(flat_tau)
        update_economic_variables(unflattened_tau, previous_tau)
        update_hats(unflattened_tau, var.t, var.pi)  # Add this line
        return gov_obj(unflattened_tau, j)

    def input_constraints(flat_tau):
        unflattened_tau = unflatten_dict(flat_tau)
        update_economic_variables(unflattened_tau, previous_tau)
        update_hats(unflattened_tau, var.t, var.pi)  # Add this line
        return constraints(unflattened_tau, j)
    
    constraint = input_constraints(initial_tau)
    
    bounds = Bounds(1, 2)
    result = minimize(
        importer_gov_obj,
        initial_tau,
        method='trust-constr',
        bounds=bounds,
        constraints = constraint,
        options={'disp': True, 'maxiter': 10000, 'gtol': 1e-10, 'xtol':1e-10}
    )
    print("MINIMIZE : ", result.success, result.message)
    final_result = np.clip(result.x, 1,2)
    optimized_tau = unflatten_dict(final_result)
    optimized_g = -(result.fun)

    return optimized_tau, optimized_g

tariff_history = {i: {j: {industry: [] for industry in var.industries} for j in var.countries if j != i} for i in var.countries}
welfare_history = {j: [] for j in var.countries} 

iteration = 100
#tolerance = 1e-50
previous_tau = var.tau.copy()
# Perform 100 iterations
for iter in range(iteration):
    print(f"Iteration {iter + 1}")
    print(var.tau)

    previous_tau = var.tau.copy()
    # Store the current state of the economic variables
    temp_p = var.p_ijs.copy() 
    temp_T = var.T.copy()
    temp_pi = var.pi.copy()

    # Iterate over each importing country and optimize
    
    
    for j in var.countries:
    # 최적화 수행
        print(f"Optimizing tariffs for imports into country: {j}")
        optimized_tau, optimized_g = optimize_for_importer(j)

        welfare_history[j].append(optimized_g)

        for i in var.countries:
            if i != j:
                for s in var.industries:
                    var.tau[i][j][s] = optimized_tau[i][j][s]
                    tariff_history[i][j][s].append(max(optimized_tau[i][j][s], 1e-10))

        # 변화율 계산
    # average_change = calculate_average_change(var.tau, previous_tau)
    # print(f"Average change in tariffs: {average_change}")

    # # 변화율이 tolerance 이하이면 반복 종료
    # if average_change < tolerance:
    #     print("Tariff changes are below the tolerance threshold. Stopping iteration.")
    #     break

    # 이전 tau 갱신
    previous_tau = var.tau.copy()


    # 모든 국가에 대해 업데이트 수행
    update_economic_variables(var.tau, previous_tau)
    update_hats(var.tau, var.t, var.pi)

    var.fill_pi()
    var.fill_gamma()
    var.fill_alpha()

    # Call welfare_change with updated delta values
    #print("welfare change effect of iteration", (iter+1), welfare_change())
    print("\nCorresponding t values:")
    for i in var.countries:
        print(f"\nt values for {i} as the home country:")
        df_t = pd.DataFrame({j: {s: var.t[i][j][s] for s in var.industries} for j in var.countries if j != i})
        print(df_t)

    # Print the current state of var.tau
    print("Nash Tariffs (tau) after iteration", iter + 1)
    for i in var.countries:
        print(f"\nTariffs for {i} as the home country:")
        df_tau = pd.DataFrame({j: {s: var.tau[i][j][s] for s in var.industries} for j in var.countries if j != i})
        print(df_tau)

    # Open the file in binary write mode and use pickle to dump the data
    with open("tariff_history.pkl", 'wb') as file:
        pickle.dump(tariff_history, file)
    with open("welfare_history.pkl", 'wb') as file:
        pickle.dump(welfare_history, file)

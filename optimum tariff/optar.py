# to check if actual tariffs get calculated
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import pickle

# file_path = 'tau_ijs.pickle'

# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# print(data)

# Define countries and industries
countries = ['China', 'Korea', 'Japan', 'USA', 'Germany']
industries = ['gim', 'steel', 'semi', 'car']
num_countries = len(countries)
num_industries = len(industries)

# Initialize dictionaries with varied values
pol_econ = {country: {industry: np.random.rand() for industry in industries} for country in countries}
x = {country: {industry: np.random.rand() for industry in industries} for country in countries}

# used when calculating welfare
p_is = {
    'China': {
        'Germany': {'gim': 11.113849080349386, 'steel': 1.0582132952603989, 'semi': 0, 'semi_memory': 0.8749017013649593, 'car': 22347.52148025721}, 
        'Japan': {'gim': 4.702912149549132, 'steel': 0.6545231939214761, 'semi': 0, 'semi_memory': 1.3782939026477783, 'car': 28876.362273677052}, 
        'Korea': {'gim': 1.9437738767665877, 'steel': 0.6248022737655794, 'semi': 0, 'semi_memory': 3.2280803660391775, 'car': 27254.069456311554}, 
        'USA': {'gim': 7.338449268069469, 'steel': 1.2736582321419347, 'semi': 0, 'semi_memory': 1.4443405822490798, 'car': 6412.672705308079}}, 
    'Germany': {
        'China': {'gim': 38.2548102189781, 'steel': 1.380815010542162, 'semi': 0, 'semi_memory': 0.2344066978065146, 'car': 67597.35761649995}, 
        'Japan': {'steel': 1.8565627953727868, 'semi': 0, 'semi_memory': 0.8579526696580784, 'car': 43312.870251408094}, 
        'Korea': {'gim': 193.40293333333332, 'steel': 2.0382712213649143, 'semi': 0, 'semi_memory': 3.0795445270513784, 'car': 53270.511687146936}, 
        'USA': {'gim': 144.169, 'steel': 0.9205237153804969, 'semi': 0, 'semi_memory': 1.1508868485166053, 'car': 63936.47417704994}}, 
    'Japan': {
        'China': {'gim': 8.886296287788753, 'steel': 0.7661557893436499, 'semi': 0, 'semi_memory': 1.9214838065500246, 'car': 34435.358268213306}, 
        'Germany': {'gim': 30.806603908794788, 'steel': 1.009475489607891, 'semi': 0, 'semi_memory': 5.56617554347278, 'car': 24246.767236564578}, 
        'Korea': {'gim': 11.861566270365477, 'steel': 0.6452938452728846, 'semi': 0, 'semi_memory': 2.576506085798766, 'car': 25945.71517654626}, 
        'USA': {'gim': 37.98874077567761, 'steel': 0.6924101202128765, 'semi': 0, 'semi_memory': 5.473356193407144, 'car': 27683.734550894653}}, 
    'USA': {
        'China': {'steel': 2.793492934160387, 'semi': 0, 'semi_memory': 1.8820447349060032, 'car': 59753.32634173833, 'gim': 202.66666666666666}, 
        'Germany': {'gim': 10.990251530053763, 'steel': 0.9743858548405673, 'semi': 0, 'semi_memory': 20.99480847346542, 'car': 45214.10895838236}, 
        'Japan': {'gim': 12.888821246314357, 'steel': 3.19057747932092, 'semi': 0, 'semi_memory': 1.4593678187228334, 'car': 51928.2106433143}, 
        'Korea': {'gim': 17.13888888888889, 'steel': 3.5832969128690055, 'semi': 0, 'semi_memory': 3.497620193153412, 'car': 47727.025701300736}}, 
    'Korea': {
        'China': {'gim': 11.346897999992917, 'steel': 0.8095350234608575, 'semi': 0, 'semi_memory': 2.9933717074944526, 'car': 24573.867215686274}, 
        'Germany': {'gim': 12.498771320251953, 'steel': 1.0443284707421892, 'semi': 0, 'semi_memory': 0, 'car': 31550.779572362917}, 
        'Japan': {'gim': 10.377253137100842, 'steel': 0.760506783231565, 'semi': 0, 'semi_memory': 2.1596662328607987, 'car': 18812.184776746457}, 
        'USA': {'gim': 16.36964135872177, 'steel': 0.9889283307131168, 'semi': 0, 'semi_memory': 6.354314554989172, 'car': 24121.347339170934}
        }            
    }

P_j = {'Germany':131.8924482,
        'China':132.2291519,
        'Japan':111.3640359,
        'Korea':129.1901759,
        'USA':139.7357936
}



# tau = {i: {j: {industry: np.random.rand() + 1 for industry in industries} for j in countries} for i in countries}
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


sigma = {'gim': 3.57, 'steel': 4.0, 'semi': 2.5, 'car': 1.8}

t = {'Korea': {'USA': {'gim': 0, 'steel': 0, 'semi': 0, 'car': 0}, 
               'China': {'gim': 0.08, 'steel': 0.005, 'semi': 0.01, 'car': 0.04}, 
               'Japan': {'gim': 0.4, 'steel': 0, 'semi': 0, 'car': 0}, 
               'Germany': {'gim': 0, 'steel': 0, 'semi': 0, 'car': 0}}, 
    'USA': {'Korea': {'gim': 0, 'steel': 0, 'semi': 0, 'car': 0}, 
            'China': {'gim': 0.2, 'steel': 0.05, 'semi': 0, 'car': 0.085}, 
            'Japan': {'gim': 0.4, 'steel': 0, 'semi': 0, 'car': 0}, 
            'Germany': {'gim': 0, 'steel': 0, 'semi': 0, 'car': 0.03}}, 
    'China': {'Korea': {'gim': 0.18, 'steel': 0, 'semi': 0, 'car': 0.059}, 
              'USA': {'gim': 0, 'steel': 0, 'semi': 0.25, 'car': 0.2275}, 
              'Japan': {'gim': 0.4, 'steel': 0, 'semi': 0, 'car': 0}, 
              'Germany': {'gim': 0, 'steel': 0.359, 'semi': 0, 'car': 0.03}}, 
    'Japan': {'Korea': {'gim': 0.2, 'steel': 0, 'semi': 0, 'car': 0.065}, 
              'USA': {'gim': 0, 'steel': 0, 'semi': 0, 'car': 0.0212}, 
              'China': {'gim': 0.175, 'steel': 0.044, 'semi': 0, 'car': 0.077}, 
              'Germany': {'gim': 0, 'steel': 0, 'semi': 0, 'car': 0}}, 
    'Germany': {'Korea': {'gim': 0, 'steel': 0, 'semi': 0, 'car': 0}, 
                'USA': {'gim': 0, 'steel': 0, 'semi': 0, 'car': 0.0212}, 
                'Japan': {'gim': 0.4, 'steel': 0, 'semi': 0, 'car': 0}, 
                'China': {'gim': 0.2, 'steel': 0.05, 'semi': 0, 'car': 0.085}}}

#계산함
pi = {country: {industry: 0 for industry in industries} for country in countries}
alpha = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}
gamma = {i: {j: {industry: 0 for industry in industries} for j in countries} for i in countries}


# optimum tariff only has one game. The initial value becomes 1
pi_hat = {country: {industry: 1 for industry in industries} for country in countries}
tau_hat = {i: {j: {industry: 1 for industry in industries} for j in countries} for i in countries}
t_hat = {i: {j: {industry: 1 for industry in industries} for j in countries} for i in countries}

# w_hat = {country: 1 for country in countries}
w = {'Korea':16.64, 'USA':33.7, 'China':6.21, 'Japan':12.33, 'Germany':25.34}


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

gamma_denom = {j: {industry: 0 for industry in industries} for j in countries}

def fill_gamma_denom():
    for j in countries:
        for s in industries:
            # gamma_denom[j][s] = 0
            for m in countries:
                if m != j:
                    gamma_denom[j][s] += tau[m][j][s] * T[m][j][s]
                

def fill_gamma():
    fill_gamma_denom()
    for i in countries:
        for j in countries:
            for s in industries:
                if i != j:
                    gamma[i][j][s] = tau[i][j][s] * T[i][j][s] / gamma_denom[j][s]
                

fill_gamma()

def fill_pi():
    for j in countries:
        for s in industries:
            for i in countries:
                if i != j:
                    pi[j][s] += 1/sigma[s] * T[i][j][s]

fill_pi()

alpha_denom = {j: {industry: 0 for industry in industries} for j in countries}

def fill_alpha_denom():
    for i in countries:
        for s in industries:
            for n in countries:
                if i != n:
                    alpha_denom[i][s] += T[i][n][s]            

def fill_alpha():
    fill_alpha_denom()
    for i in countries:
        for j in countries:
            for s in industries:
                if i != j:
                    alpha[i][j][s] += T[i][j][s] / alpha_denom[i][s]               

fill_alpha()

# Welfare function
def welfare(j, s):
    return x[j][s] + P_j[j]

# Government objective function for country j
def gov_obj(tau_js, j):
    # tau_copy = {i: {k: {industry: tau[i][k][industry] for industry in industries} for k in countries} for i in countries}
    tau_copy = {i: {k: {industry: 0 for industry in industries} for k in countries} for i in countries}
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

def complicated(j):
    total = 0
    for i in countries:
        for s in industries:
            if i != j:
                total += (t[i][j][s] * T[i][j][s] / x2(j) * t_hat[i][j][s] * 
                        (eq_12(j, s) ** (sigma[s] - 1)) * (tau_hat[i][j][s] ** -sigma[s]) + (pi[j][s] / x2(j) * pi_hat[j][s]))
    return total

def term3(j):
    total = 0
    for s in industries:
        for i in countries:
            if i != j:
                total += (pi[j][s] / x2(j) * pi_hat[j][s]) * alpha[j][i][s] * (tau_hat[j][i][s] ** -sigma[s]) * (w[i] ** (1 - sigma[s])) * (eq_12(i, s) ** (sigma[s] - 1))
    return total

def eq_13(j):
    term1 = wL(j) / x2(j)
    term2 = complicated(j)
    term3 = 0

    for s in industries:
        term3 += pi[j][s] / x2(j) * pi_hat[j][s]
    
    sum = (term1) / (1 - term2 - term3)

    return term1 + term2 + term3 - 1  # Constraint to be equal to 1

# Constraint 3 for country i and industry s
def eq_10(i, s):
    total = 0
    for j in countries:
        if i != j:
            total += (alpha[i][j][s] * (tau_hat[i][j][s] ** -sigma[s]) * 
                    (w[i] ** (1 - sigma[s])) * (eq_12(j, s) ** (sigma[s] - 1)) * eq_13(j))
    return total - 1  # Constraint to be equal to 1

# Constraints as a list for country j
def constraints(tau_js, j):
    tau_copy = {i: {k: {industry: 0 for industry in industries} for k in countries} for i in countries}
    for i, industry in enumerate(industries):
        for k, country in enumerate(countries):
            if country != j:
                tau_copy[country][j][industry] = tau_js[i * num_countries + k]
                # tau_copy[country][j][industry] = tau_js[j][industry]
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
        # initial_tau_js = tau[i]
        result = minimize(gov_obj, initial_tau_js, args=(j,), constraints=constraints(initial_tau_js, j))
        for k, industry in enumerate(industries):
            for m, country in enumerate(countries):
                if country != i:
                    optimal_taus[country][industry] = result.x[k * num_countries + m]
    
    print(f"Optimal tariffs for {i} as the home country:")
    print(pd.DataFrame(optimal_taus))
    print("\n")

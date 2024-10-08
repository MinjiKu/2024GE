# to check if actual tariffs get calculated
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt

# Define countries and industries
countries = ['China', 'Korea', 'Japan', 'USA', 'Germany']
industries = ['steel', 'semi', 'car']
num_countries = len(countries)
num_industries = len(industries)
epsilon = 1e-10 # a very small number

# Initialize dictionaries with varied values
pol_econ = {}
for country in countries:
    random_values = np.random.rand(num_industries)
    normalized_values = (random_values / random_values.sum()) * 3
    pol_econ[country] = {industry: normalized_values[i] for i, industry in enumerate(industries)}

# x = {country: {industry: np.random.rand() for industry in industries} for country in countries}

def coop_lambda():
    # Loop through each country and set the industry values to 1
    for country in countries:
        pol_econ[country] = {industry: 1 for industry in industries}

x = { #우선 김은 임의로 100으로 채움
         #단위는 billion USD
    'Korea':{'steel':65.85, 'semi':19.8, 'car':188.856},
    'USA': {'steel':4.14, 'semi':80.5, 'car':768.0},
    'China': {'steel':421.18, 'semi':193.6, 'car':772},
    'Japan':{'steel':58.92, 'semi':51.9, 'car':476.784},
    'Germany':{'steel':44.50, 'semi':41.2, 'car': 614.313}
}

# used when calculating welfare
p_ijs = {
    'China': {
        'Germany': {'steel': 1.0582132952603989, 'semi': 0.8749017013649593, 'car': 22347.52148025721}, 
        'Japan': {'steel': 0.6545231939214761, 'semi': 1.3782939026477783, 'car': 28876.362273677052}, 
        'Korea': {'steel': 0.6248022737655794, 'semi': 3.2280803660391775, 'car': 27254.069456311554}, 
        'USA': {'steel': 1.2736582321419347, 'semi': 1.4443405822490798, 'car': 6412.672705308079}}, 
    'Germany': {
        'China': {'steel': 1.380815010542162, 'semi': 0.2344066978065146, 'car': 67597.35761649995}, 
        'Japan': {'steel': 1.8565627953727868, 'semi': 0.8579526696580784, 'car': 43312.870251408094}, 
        'Korea': {'steel': 2.0382712213649143,'semi': 3.0795445270513784, 'car': 53270.511687146936}, 
        'USA': {'steel': 0.9205237153804969, 'semi': 1.1508868485166053, 'car': 63936.47417704994}}, 
    'Japan': {
        'China': {'steel': 0.7661557893436499,  'semi': 1.9214838065500246, 'car': 34435.358268213306}, 
        'Germany': {'steel': 1.009475489607891,  'semi': 5.56617554347278, 'car': 24246.767236564578}, 
        'Korea': {'steel': 0.6452938452728846,  'semi': 2.576506085798766, 'car': 25945.71517654626}, 
        'USA': {'steel': 0.6924101202128765, 'semi': 5.473356193407144, 'car': 27683.734550894653}}, 
    'USA': {
        'China': {'steel': 2.793492934160387, 'semi': 1.8820447349060032, 'car': 59753.32634173833, 'gim': 202.66666666666666}, 
        'Germany': {'steel': 0.9743858548405673, 'semi': 20.99480847346542, 'car': 45214.10895838236}, 
        'Japan': {'steel': 3.19057747932092, 'semi': 1.4593678187228334, 'car': 51928.2106433143}, 
        'Korea': {'steel': 3.5832969128690055, 'semi': 3.497620193153412, 'car': 47727.025701300736}}, 
    'Korea': {
        'China': {'steel': 0.8095350234608575, 'semi': 2.9933717074944526, 'car': 24573.867215686274}, 
        'Germany': {'steel': 1.0443284707421892, 'semi': 3.0795445270513784, 'car': 31550.779572362917}, 
        'Japan': {'steel': 0.760506783231565, 'semi': 2.1596662328607987, 'car': 18812.184776746457}, 
        'USA': {'steel': 0.9889283307131168, 'semi': 6.354314554989172, 'car': 24121.347339170934}
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
        'Korea': {'steel': 1, 'semi': 1, 'car': 1.059},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1, 'semi': 1.25, 'car': 1.2275},
        'Germany': {'steel': 1.359, 'semi': 1, 'car': 1.03}
    },
    'Korea': {
        'China': {'steel': 1.005, 'semi': 1.01, 'car': 1.04},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1, 'semi': 1, 'car': 1},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1}
    },
    'Japan': {
        'China': {'steel': 1.044, 'semi': 1, 'car': 1.077},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1.065},
        'USA': {'steel': 1, 'semi': 1, 'car': 1.0212},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1}
    },
    'USA': {
        'China': {'steel': 1.05, 'semi': 1, 'car': 1.085},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1.03}
    },
    'Germany': {
        'China': {'steel': 1.05, 'semi': 1, 'car': 1.085},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1, 'semi': 1, 'car': 1.0212}
    }
}
# # iteration 2
# tau2 = {
#     'China': {
#         'Korea': {'steel': 1.05, 'semi': 1.05, 'car': 1.109},
#         'Japan': {'steel': 1.05, 'semi': 1.05, 'car': 1.05},
#         'USA': {'steel': 1, 'semi': 1.00, 'car': 1.20},
#         'Germany': {'steel': 1.409, 'semi': 1.05, 'car': 1.08}
#     },
#     'Korea': {
#         'China': {'steel': 1.00, 'semi': 1.0, 'car': 1.0},
#         'Japan': {'steel': 1.05, 'semi': 1.05, 'car': 1.05},
#         'USA': {'steel': 1, 'semi': 1, 'car': 1},
#         'Germany': {'steel': 1.05, 'semi': 1.05, 'car': 1.05}
#     },
#     'Japan': {
#         'China': {'steel': 1.0, 'semi': 1, 'car': 1.02},
#         'Korea': {'steel': 1.05, 'semi': 1.05, 'car': 1.115},
#         'USA': {'steel': 1, 'semi': 1, 'car': 1.0002},
#         'Germany': {'steel': 1.05, 'semi': 1.05, 'car': 1.05}
#     },
#     'USA': {
#         'China': {'steel': 1.00, 'semi': 1, 'car': 1.035},
#         'Korea': {'steel': 1.05, 'semi': 1.05, 'car': 1.05},
#         'Japan': {'steel': 1.05, 'semi': 1.05, 'car': 1.05},
#         'Germany': {'steel': 1.05, 'semi': 1.05, 'car': 1.08}
#     },
#     'Germany': {
#         'China': {'steel': 1.00, 'semi': 1, 'car': 1.035},
#         'Korea': {'steel': 1.05, 'semi': 1.05, 'car': 1.05},
#         'Japan': {'steel': 1.05, 'semi': 1.05, 'car': 1.05},
#         'USA': {'steel': 1, 'semi': 1, 'car': 1.0012}
#     }
# }
# # iteration 3
# tau3 = {
#     'China': {
#         'Korea': {'steel': 1.1, 'semi': 1.1, 'car': 1.159},
#         'Japan': {'steel': 1.15, 'semi': 1.15, 'car': 1.15},
#         'USA': {'steel': 1, 'semi': 1.00, 'car': 1.00},
#         'Germany': {'steel': 1.459, 'semi': 1.1, 'car': 1.13}
#     },
#     'Korea': {
#         'China': {'steel': 1.00, 'semi': 1.0, 'car': 1.0},
#         'Japan': {'steel': 1.1, 'semi': 1.1, 'car': 1.1},
#         'USA': {'steel': 1, 'semi': 1, 'car': 1},
#         'Germany': {'steel': 1.15, 'semi': 1.15, 'car': 1.15}
#     },
#     'Japan': {
#         'China': {'steel': 1.0, 'semi': 1, 'car': 1.00},
#         'Korea': {'steel': 1.1, 'semi': 1.1, 'car': 1.165},
#         'USA': {'steel': 1, 'semi': 1, 'car': 1.000},
#         'Germany': {'steel': 1.1, 'semi': 1.1, 'car': 1.1}
#     },
#     'USA': {
#         'China': {'steel': 1.00, 'semi': 1, 'car': 1.0},
#         'Korea': {'steel': 1.1, 'semi': 1.1, 'car': 1.1},
#         'Japan': {'steel': 1.1, 'semi': 1.1, 'car': 1.1},
#         'Germany': {'steel': 1.1, 'semi': 1.1, 'car': 1.13}
#     },
#     'Germany': {
#         'China': {'steel': 1.00, 'semi': 1, 'car': 1.0},
#         'Korea': {'steel': 1.1, 'semi': 1.1, 'car': 1.1},
#         'Japan': {'steel': 1.1, 'semi': 1.1, 'car': 1.1},
#         'USA': {'steel': 1, 'semi': 1, 'car': 1.00}
#     }
# }
# # iteration 4
# tau4 = {
#     'China': {
#         'Korea': {'steel': 1.15, 'semi': 1.15, 'car': 1.209},
#         'Japan': {'steel': 1.20, 'semi': 1.2, 'car': 1.2},
#         'USA': {'steel': 1, 'semi': 1.00, 'car': 1.00},
#         'Germany': {'steel': 1.509, 'semi': 1.15, 'car': 1.18}
#     },
#     'Korea': {
#         'China': {'steel': 1.00, 'semi': 1.0, 'car': 1.0},
#         'Japan': {'steel': 1.15, 'semi': 1.15, 'car': 1.15},
#         'USA': {'steel': 1, 'semi': 1, 'car': 1},
#         'Germany': {'steel': 1.2, 'semi': 1.2, 'car': 1.2}
#     },
#     'Japan': {
#         'China': {'steel': 1.0, 'semi': 1, 'car': 1.00},
#         'Korea': {'steel': 1.15, 'semi': 1.15, 'car': 1.215},
#         'USA': {'steel': 1, 'semi': 1, 'car': 1.000},
#         'Germany': {'steel': 1.15, 'semi': 1.15, 'car': 1.15}
#     },
#     'USA': {
#         'China': {'steel': 1.00, 'semi': 1, 'car': 1.0},
#         'Korea': {'steel': 1.15, 'semi': 1.15, 'car': 1.15},
#         'Japan': {'steel': 1.15, 'semi': 1.15, 'car': 1.15},
#         'Germany': {'steel': 1.15, 'semi': 1.15, 'car': 1.18}
#     },
#     'Germany': {
#         'China': {'steel': 1.00, 'semi': 1, 'car': 1.0},
#         'Korea': {'steel': 1.15, 'semi': 1.15, 'car': 1.15},
#         'Japan': {'steel': 1.15, 'semi': 1.15, 'car': 1.15},
#         'USA': {'steel': 1, 'semi': 1, 'car': 1.00}
#     }
# }

tau = {
    'China': {
        'Korea': {'steel': 1, 'semi': 1, 'car': 1.059},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1, 'semi': 1.25, 'car': 1.2275},
        'Germany': {'steel': 1.359, 'semi': 1, 'car': 1.03}
    },
    'Korea': {
        'China': {'steel': 1.005, 'semi': 1.01, 'car': 1.04},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1, 'semi': 1, 'car': 1},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1}
    },
    'Japan': {
        'China': {'steel': 1.044, 'semi': 1, 'car': 1.077},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1.065},
        'USA': {'steel': 1, 'semi': 1, 'car': 1.0212},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1}
    },
    'USA': {
        'China': {'steel': 1.05, 'semi': 1, 'car': 1.085},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1.03}
    },
    'Germany': {
        'China': {'steel': 1.05, 'semi': 1, 'car': 1.085},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1, 'semi': 1, 'car': 1.0212}
    }
}
# iteration 2
tau2 = {
    'China': {
        'Korea': {'steel': 1, 'semi': 1, 'car': 1.029},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1.05, 'semi': 1.3, 'car': 1.2775},
        'Germany': {'steel': 1.319, 'semi': 1, 'car': 1.00}
    },
    'Korea': {
        'China': {'steel': 1.055, 'semi': 1.06, 'car': 1.09},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1.05, 'semi': 1.05, 'car': 1.05},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1}
    },
    'Japan': {
        'China': {'steel': 1.094, 'semi': 1.05, 'car': 1.127},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1.035},
        'USA': {'steel': 1.05, 'semi': 1.05, 'car': 1.0712},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1}
    },
    'USA': {
        'China': {'steel': 1.1, 'semi': 1.05, 'car': 1.135},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1.00}
    },
    'Germany': {
        'China': {'steel': 1.1, 'semi': 1.05, 'car': 1.135},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1.05, 'semi': 1.05, 'car': 1.0712}
    }
}
# iteration 3
tau3 = {
    'China': {
        'Korea': {'steel': 1, 'semi': 1, 'car': 1.0},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1.1, 'semi': 1.35, 'car': 1.3275},
        'Germany': {'steel': 1.279, 'semi': 1, 'car': 1.00}
    },
    'Korea': {
        'China': {'steel': 1.105, 'semi': 1.11, 'car': 1.14},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1.1, 'semi': 1.1, 'car': 1.1},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1}
    },
    'Japan': {
        'China': {'steel': 1.144, 'semi': 1.1, 'car': 1.177},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1.005},
        'USA': {'steel': 1.1, 'semi': 1.1, 'car': 1.1212},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1}
    },
    'USA': {
        'China': {'steel': 1.15, 'semi': 1.1, 'car': 1.185},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1.00}
    },
    'Germany': {
        'China': {'steel': 1.15, 'semi': 1.1, 'car': 1.185},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1.1, 'semi': 1.1, 'car': 1.1212}
    }
}
# iteration 4
tau4 = {
    'China': {
        'Korea': {'steel': 1, 'semi': 1, 'car': 1.0},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1.15, 'semi': 1.4, 'car': 1.3775},
        'Germany': {'steel': 1.239, 'semi': 1, 'car': 1.00}
    },
    'Korea': {
        'China': {'steel': 1.155, 'semi': 1.16, 'car': 1.19},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1.15, 'semi': 1.15, 'car': 1.15},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1}
    },
    'Japan': {
        'China': {'steel': 1.194, 'semi': 1.15, 'car': 1.227},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1.00},
        'USA': {'steel': 1.15, 'semi': 1.15, 'car': 1.1712},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1}
    },
    'USA': {
        'China': {'steel': 1.2, 'semi': 1.15, 'car': 1.235},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1.00}
    },
    'Germany': {
        'China': {'steel': 1.2, 'semi': 1.15, 'car': 1.235},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1.15, 'semi': 1.15, 'car': 1.1712}
    }
}

# iteration 5
tau5 = {
    'China': {
        'Korea': {'steel': 1, 'semi': 1, 'car': 1.0},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1.2, 'semi': 1.45, 'car': 1.4275},
        'Germany': {'steel': 1.199, 'semi': 1, 'car': 1.00}
    },
    'Korea': {
        'China': {'steel': 1.205, 'semi': 1.21, 'car': 1.24},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1.2, 'semi': 1.2, 'car': 1.2},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1}
    },
    'Japan': {
        'China': {'steel': 1.244, 'semi': 1.2, 'car': 1.277},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1.00},
        'USA': {'steel': 1.2, 'semi': 1.2, 'car': 1.2212},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1}
    },
    'USA': {
        'China': {'steel': 1.25, 'semi': 1.2, 'car': 1.285},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1.00}
    },
    'Germany': {
        'China': {'steel': 1.25, 'semi': 1.2, 'car': 1.285},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1.2, 'semi': 1.2, 'car': 1.2212}
    }
}

# iteration 5
tau5 = {
    'China': {
        'Korea': {'steel': 1, 'semi': 1, 'car': 1.0},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1.2, 'semi': 1.45, 'car': 1.4275},
        'Germany': {'steel': 1.199, 'semi': 1, 'car': 1.00}
    },
    'Korea': {
        'China': {'steel': 1.205, 'semi': 1.21, 'car': 1.24},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1.2, 'semi': 1.2, 'car': 1.2},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1}
    },
    'Japan': {
        'China': {'steel': 1.244, 'semi': 1.2, 'car': 1.277},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1.00},
        'USA': {'steel': 1.2, 'semi': 1.2, 'car': 1.2212},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1}
    },
    'USA': {
        'China': {'steel': 1.25, 'semi': 1.2, 'car': 1.285},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'Germany': {'steel': 1, 'semi': 1, 'car': 1.00}
    },
    'Germany': {
        'China': {'steel': 1.25, 'semi': 1.2, 'car': 1.285},
        'Korea': {'steel': 1, 'semi': 1, 'car': 1},
        'Japan': {'steel': 1, 'semi': 1, 'car': 1},
        'USA': {'steel': 1.2, 'semi': 1.2, 'car': 1.2212}
    }
}

tau_temp = {country: {partner: {product: value * 100 for product, value in products.items()}
               for partner, products in partners.items()}
     for country, partners in tau.items()}


sigma = {'gim': 3.57, 'steel': 4.0, 'semi': 2.5, 'car': 1.8}

t = {'Korea': {'USA': {'steel': 1e-10, 'semi': 1e-10, 'car': 1e-10}, 
               'China': {'steel': 0.005, 'semi': 0.01, 'car': 0.04}, 
               'Japan': {'steel': 1e-10, 'semi': 1e-10, 'car': 1e-10}, 
               'Germany': {'steel': 1e-10, 'semi': 1e-10, 'car': 1e-10}}, 
    'USA': {'Korea': {'steel': 1e-10, 'semi': 1e-10, 'car': 1e-10}, 
            'China': {'steel': 0.05, 'semi': 1e-10, 'car': 0.085}, 
            'Japan': {'steel': 1e-10, 'semi': 1e-10, 'car': 1e-10}, 
            'Germany': {'steel': 1e-10, 'semi': 1e-10, 'car': 0.03}}, 
    'China': {'Korea': {'steel': 1e-10, 'semi': 1e-10, 'car': 0.059}, 
              'USA': {'steel': 1e-10, 'semi': 0.25, 'car': 0.2275}, 
              'Japan': {'steel': 1e-10, 'semi': 1e-10, 'car': 1e-10}, 
              'Germany': {'steel': 0.359, 'semi': 1e-10, 'car': 0.03}}, 
    'Japan': {'Korea': {'steel': 1e-10, 'semi': 1e-10, 'car': 0.065}, 
              'USA': {'steel': 1e-10, 'semi': 1e-10, 'car': 0.0212}, 
              'China': {'steel': 0.044, 'semi': 1e-10, 'car': 0.077}, 
              'Germany': {'steel': 1e-10, 'semi': 1e-10, 'car': 1e-10}}, 
    'Germany': {'Korea': {'steel': 1e-10, 'semi': 1e-10, 'car': 1e-10}, 
                'USA': {'steel': 1e-10, 'semi': 1e-10, 'car': 0.0212}, 
                'Japan': {'steel': 1e-10, 'semi': 1e-10, 'car': 1e-10}, 
                'China': {'steel': 0.05, 'semi': 1e-10, 'car': 0.085}}}

# t = {country: {partner: {product: value * 100 for product, value in products.items()}
#                for partner, products in partners.items()}
#      for country, partners in t_temp.items()}


#계산함
pi = {country: {industry: 0 for industry in industries} for country in countries}
# Initialize gamma and alpha
gamma = {i: {j: {industry: 0 for industry in industries} for j in countries if j != i} for i in countries}
alpha = {i: {j: {industry: 0 for industry in industries} for j in countries if j != i} for i in countries}

#계산이 거듭함에도 factual_들은 원래 값을 보존
factual_tau = tau.copy()
factual_t = t.copy()

# optimum tariff only has one game. The initial value becomes 1
pi_hat = {country: {industry: 1 for industry in industries} for country in countries}
tau_hat = {i: {j: {industry: 1 for industry in industries} for j in countries if j != i} for i in countries}
t_hat = {i: {j: {industry: 1 for industry in industries} for j in countries if j != i} for i in countries}

#뒤에서 계산해서 채워야 함
P_hat = {j: {s: 1.0 for s in industries} for j in countries}
X_hat = {j: 1.0 for j in countries} 

# w_hat = {country: 1 for country in countries}
w = {'Korea':16.64, 'USA':33.7, 'China':6.21, 'Japan':12.33, 'Germany':25.34}


T = {
    'China': {'Germany': {'steel': 582083.0, 'semi': 119516177.0, 'car': 1833609424.0}, 
              'Japan': {'steel': 83818503.0, 'semi': 535757854.0, 'car': 676593127.0}, 
              'Korea': {'steel': 1803944738.0, 'semi': 15219366037.0, 'car': 1212097485.0}, 
              'USA': {'steel': 1358591.0, 'semi': 341810421.0, 'car': 2638478640.0}
            }, 
    'Germany': {'China': {'steel': 5201023.164, 'semi': 19331662.139, 'car': 16512164176.907999}, 
                'Japan': {'steel': 40160.479, 'semi': 693421.891, 'car': 4282380533.395}, 
                'Korea': {'steel': 681557.131, 'semi': 3842475.101, 'car': 6468871825.364}, 
                'USA': {'steel': 51998280.381, 'semi': 12024057.385, 'car': 28218169963.792}
            }, 
    'Japan': {'China': {'steel': 777891701.735, 'semi': 4241031727.742, 'car': 6714475489.161}, 
              'Germany': {'steel': 12493288.74, 'semi': 40437523.308, 'car': 2518119540.9589996}, 
              'Korea': {'steel': 1873216558.145, 'semi': 20384960.593, 'car': 746276605.623}, 
              'USA': {'steel': 141884446.045, 'semi': 57715066.618, 'car': 43064344998.029}
            }, 
    'USA': {'China': {'steel': 487349.0,'semi': 41034340.0, 'car': 6554910660.0}, 
            'Germany': {'steel': 75461.0, 'semi': 18155007.0, 'car': 9157689622.0}, 
            'Japan': {'steel': 1199499.0, 'semi': 14758843.0, 'car': 1277318495.0}, 
            'Korea': {'steel': 180376.0,'semi': 15309944.0, 'car': 2740915359.0}
            }, 
    'Korea': {'China': {'steel': 543488181.0,  'semi': 34451077419.0, 'car': 313316807.0}, 
              'Germany': {'steel': 52080498.965, 'semi': 156173609.446, 'car': 11664726.954}, 
              'Japan': {'steel': 921335767.82, 'semi': 419536125.294, 'car': 495404.448}, 
              'USA': {'steel': 581315367.0, 'semi': 236493837.0, 'car': 883492635.0}
            }
}

L_js = {
    'Korea':{'steel':1.13, 'semi':0.34, 'car':3.25},
    'USA': {'steel':0.03, 'semi':0.50, 'car':4.80},
    'China': {'steel':18.46, 'semi':8.49, 'car':33.84},
    'Japan':{'steel':0.97, 'semi':0.85, 'car':7.84},
    'Germany':{'steel':0.44, 'semi':0.41, 'car': 6.12}
}

L_j = {'Korea':29.5, 'USA':171, 'China':780, 'Japan':69.3, 'Germany':44.4} #million 

#demand elasticity
de = {'steel':0.5, 'semi':0.5, 'car':1.5}

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
                    gamma_value = tau[i][j][s] * T[i][j][s] / gamma_denom[j][s]
                    gamma[i][j][s] = max(gamma_value, epsilon)
                

fill_gamma()
# print("gamma:")
# print(gamma)
# print("\n")
# print("gamma_denom: ")
# print(gamma_denom)
# print("\n")

def fill_pi():
    for j in countries:
        for s in industries:
            for i in countries:
                if i != j:
                    pi_value = 1 / sigma[s] * T[i][j][s]
                    pi[j][s] += max(pi_value, epsilon)
fill_pi()
factual_pi = pi.copy() #factual pi 보존

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
            if i == j:
                continue  # Skip the case where i == j
            for s in industries:
                alpha_value = T[i][j][s] / alpha_denom[i][s]
                alpha[i][j][s] = max(alpha_value, epsilon)  # Ensure no zero values

fill_alpha()
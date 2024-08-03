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

x = { #우선 김은 임의로 100으로 채움
         #단위는 billion USD
    'Korea':{'steel':65.85, 'semi':19.8, 'car':188.856},
    'USA': {'steel':4.14, 'semi':80.5, 'car':768.0},
    'China': {'steel':421.18, 'semi':193.6, 'car':772},
    'Japan':{'steel':58.92, 'semi':51.9, 'car':476.784},
    'Germany':{'steel':44.50, 'semi':41.2, 'car': 614.313}
}

# used when calculating welfare
p_is = {
    'China': {
        'Germany': {'gim': 11.113849080349386, 'steel': 1.0582132952603989, 'semi': 0.8749017013649593, 'car': 22347.52148025721}, 
        'Japan': {'gim': 4.702912149549132, 'steel': 0.6545231939214761, 'semi': 1.3782939026477783, 'car': 28876.362273677052}, 
        'Korea': {'gim': 1.9437738767665877, 'steel': 0.6248022737655794, 'semi': 3.2280803660391775, 'car': 27254.069456311554}, 
        'USA': {'gim': 7.338449268069469, 'steel': 1.2736582321419347, 'semi': 1.4443405822490798, 'car': 6412.672705308079}}, 
    'Germany': {
        'China': {'gim': 38.2548102189781, 'steel': 1.380815010542162, 'semi': 0.2344066978065146, 'car': 67597.35761649995}, 
        'Japan': {'steel': 1.8565627953727868, 'semi': 0.8579526696580784, 'car': 43312.870251408094}, 
        'Korea': {'gim': 193.40293333333332, 'steel': 2.0382712213649143,'semi': 3.0795445270513784, 'car': 53270.511687146936}, 
        'USA': {'gim': 144.169, 'steel': 0.9205237153804969, 'semi': 1.1508868485166053, 'car': 63936.47417704994}}, 
    'Japan': {
        'China': {'steel': 0.7661557893436499,  'semi': 1.9214838065500246, 'car': 34435.358268213306}, 
        'Germany': {'steel': 1.009475489607891,  'semi': 5.56617554347278, 'car': 24246.767236564578}, 
        'Korea': {'steel': 0.6452938452728846,  'semi': 2.576506085798766, 'car': 25945.71517654626}, 
        'USA': {'steel': 0.6924101202128765, 'semi': 5.473356193407144, 'car': 27683.734550894653}}, 
    'USA': {
        'China': {'semi': 1.8820447349060032, 'car': 59753.32634173833, 'gim': 202.66666666666666}, 
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
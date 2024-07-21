import numpy as np
#%% data 정리

# practice

trade_flows = {#단위는 천 달러, Reporter별로 데이터가 다른 경우, 국제기구의 데이터 > 수출국의 데이터 순으로 신뢰함. 
    'Korea': {'USA': {'gim': , 'steel':580234, 'semi':711206, 'car':72066},
              'China': {'gim':, 'steel':454598, 'semi':17018970, 'car':16041614},
              'Japan':{'gim':, 'steel':903090, 'semi':308979, 'car':12433},
              'Germany': {'gim':, 'steel':53993, 'semi':121601, 'car':657654},},
    'USA': {'Korea': {'gim': , 'steel':95, 'semi':1092079, 'car':963233},
            'China': {'gim':,'steel':820, 'semi':8306170, 'car':9030967 },
            'Japan': {'gim': , 'steel':, 'semi':, 'car':},
            'Germany':{'gim': , 'steel':, 'semi':, 'car':}},
    'China': {'Korea': {'gim':28052,'steel':1803944, 'semi':17758308, 'car':1212067},
              'USA': {'gim':113463,'steel':1325, 'semi':2324998, 'car':2610238},
              'Japan': {'gim':306966, 'steel':83818, 'semi':3119372, 'car':2610238},
              'Germany':{'gim':23529, 'steel':582, 'semi':1636760, 'car':1833607}},
    'Japan': {'Korea': {'gim': 1329,'steel':2055349, 'semi':1921372, 'car':296440},
              'USA': {'gim':,'steel':, 'semi':, 'car':},
              'China': {'gim':, 'steel':777891, 'semi':6882555, 'car':6714475},
              'Germany':{'gim':, 'steel':12493, 'semi':370643, 'car':2518119}},
    'Germany': {'Korea': {'gim':,'steel':681, 'semi':628437, 'car':6468871},
                'USA': {'gim':,'steel':, 'semi':, 'car':},
              'Japan': {'gim':, 'steel':40, 'semi':434221, 'car':4282380},
              'China':{'gim':, 'steel':5201, 'semi':1436904, 'car':16512164}}
    
}

X = {
    'Korea':, 
    'USA': , 
    'China': , 
    'Japan':, 
    'Germany': 
}

sigma = {'gim': 3.57, 'steel': 4.0, 'semi': 2.5, 'car': 1.8}

#%%
import numpy as np
# 초기 및 최종 가격 데이터
initial_prices = {'gim': 100, 'steel': 200, 'semi': 300, 'car': 400}
final_prices = {'gim': 105, 'steel': 206, 'semi': 312, 'car': 424}

# 가격 변화를 계산
price_changes = calculate_price_changes(initial_prices, final_prices)

#%%
# Function to calculate etajs
def calculate_etajs(sigma, trade_flows):
    etajs = {}
    for industry, sigma_s in sigma.items():
        etajs[industry] = (1 / sigma_s) * np.sum([trade_flow[industry] for country in trade_flows.values() for trade_flow in country.values()])
    return etajs

# Calculate etajs
etajs = calculate_etajs(sigma, trade_flows)
                                 
#%%                           
# Calculate welfare change
def calculate_welfare_change(trade_flows, exports, price_changes, elasticities, tariff_rates, trade_flows_changes):
    terms = []
    for industry in sigma.keys():
        Tijs = np.array([trade_flow[industry] for country in trade_flows.values() for trade_flow in country.values()])
        Xj = np.array(list(exports.values()))
        dpjs = price_changes['dpjs'][industry]
        dpis = price_changes['dpis'][industry]
        etajs = elasticities[industry]
        tijs = tariff_rates[industry]
        dTijs_Tijs = trade_flows_changes['dTijs_Tijs'][industry]
        
        term1 = Tijs * (dpjs / Xj - dpis / Xj)
        term2 = etajs * (dpjs / Xj - dpjs / Xj)
        term3 = tijs * Tijs * (dTijs_Tijs / Xj - dpis / Xj)
        
        total_change = np.sum(term1 + term2 + term3)
        terms.append((term1, term2, term3, total_change))
    
    return terms
#%%

# Example data (replace these with actual values)
tariff_revenue = [100, 200]  # Example values
exports = [1000, 1500]
price_changes = {'dpjs': [0.05, 0.03], 'dpis': [0.02, 0.04]}
elasticities = {'etajs': [1.2, 0.8], 'deta_etajs': [0.01, 0.02]}
tariff_rates = [0.1, 0.15]
trade_flows = {'Tijs': [500, 600], 'dTijs_Tijs': [0.02, 0.03]}

# Calculate welfare change
welfare_change = calculate_welfare_change(tariff_revenue, exports, price_changes, elasticities, tariff_rates, trade_flows)
print(f'Percentage change in welfare: {welfare_change}')

#%% 최종
import numpy as np

# 가격 데이터를 기반으로 가격 변화를 계산하는 함수
def calculate_price_changes(initial_prices, final_prices):
    initial_prices_array = np.array(list(initial_prices.values()))
    final_prices_array = np.array(list(final_prices.values()))
    
    change_rates = (final_prices_array - initial_prices_array) / initial_prices_array
    
    industries = list(initial_prices.keys())
    price_changes = {
        'dpjs': dict(zip(industries, change_rates)),
        'dpis': dict(zip(industries, change_rates))
    }
    return price_changes

# 초기 및 최종 가격 데이터
initial_prices = {'gim': 100, 'steel': 200, 'semi': 300, 'car': 400}
final_prices = {'gim': 105, 'steel': 206, 'semi': 312, 'car': 424}

# 가격 변화를 계산
price_changes = calculate_price_changes(initial_prices, final_prices)

# 결과 출력
print(price_changes)

# Elasticities
sigma = {'gim': 3.57, 'steel': 4.0, 'semi': 2.5, 'car': 1.8}

# Trade flows data structure
trade_flows = {
    'Korea': {'USA': {'gim': 196766, 'steel': 580234, 'semi': 711206, 'car': 72066}, 
              'China': {'gim': 101299, 'steel': 454598, 'semi': 17018970, 'car': 16041614},
              'Japan': {'gim': 187401, 'steel': 903090, 'semi': 308979, 'car': 12433}, 
              'Germany': {'gim': 12194, 'steel': 53993, 'semi': 121601, 'car': 657654}},
    'USA': {'Korea': {'gim': 19076, 'steel': 95, 'semi': 1092079, 'car': 963233}, 
            'China': {'gim': 15075, 'steel': 820, 'semi': 8306170, 'car': 9030967},
            'Japan': {'gim': 20000, 'steel': 5000, 'semi': 15000, 'car': 70000}, 
            'Germany': {'gim': 23529, 'steel': 8000, 'semi': 25000, 'car': 90000}},
    'China': {'Korea': {'gim': 28052, 'steel': 1803944, 'semi': 17758308, 'car': 1212067}, 
              'USA': {'gim': 113463, 'steel': 1325, 'semi': 2324998, 'car': 2610238},
              'Japan': {'gim': 306966, 'steel': 83818, 'semi': 3119372, 'car': 2610238}, 
              'Germany': {'gim': 23529, 'steel': 582, 'semi': 1636760, 'car': 1833607}},
    'Japan': {'Korea': {'gim': 1329, 'steel': 2055349, 'semi': 1921372, 'car': 296440}, 
              'USA': {'gim': 5000, 'steel': 1000, 'semi': 12000, 'car': 80000},
              'China': {'gim': 1459, 'steel': 777891, 'semi': 6882555, 'car': 6714475}, 
              'Germany': {'gim': 738, 'steel': 12493, 'semi': 370643, 'car': 2518119}},
    'Germany': {'Korea': {'gim': 125, 'steel': 681, 'semi': 628437, 'car': 6468871}, 
                'USA': {'gim': 10000, 'steel': 5000, 'semi': 30000, 'car': 120000},
                'Japan': {'gim': 110, 'steel': 40, 'semi': 434221, 'car': 4282380}, 
                'China': {'gim': 122, 'steel': 5201, 'semi': 1436904, 'car': 16512164}}
}

# Exports data structure
exports = {
    'Korea': 500000, 
    'USA': 600000, 
    'China': 700000, 
    'Japan': 800000, 
    'Germany': 900000
}

# Trade flows changes (example values)
trade_flows_changes = {
    'dTijs_Tijs': {'gim': 0.02, 'steel': 0.03, 'semi': 0.01, 'car': 0.04}
}

# Function to calculate etajs
def calculate_etajs(sigma, trade_flows):
    etajs = {}
    for industry, sigma_s in sigma.items():
        etajs[industry] = (1 / sigma_s) * np.sum([trade_flow[industry] for country in trade_flows.values() for trade_flow in country.values()])
    return etajs

# Calculate etajs
etajs = calculate_etajs(sigma, trade_flows)

# Calculate welfare change
def calculate_welfare_change(trade_flows, exports, price_changes, elasticities, tariff_rates, trade_flows_changes):
    terms = []
    for industry in sigma.keys():
        Tijs = np.array([trade_flow[industry] for country in trade_flows.values() for trade_flow in country.values()])
        Xj = np.array(list(exports.values()))
        dpjs = price_changes['dpjs'][industry]
        dpis = price_changes['dpis'][industry]
        etajs = elasticities[industry]
        tijs = tariff_rates[industry]
        dTijs_Tijs = trade_flows_changes['dTijs_Tijs'][industry]
        
        term1 = Tijs * (dpjs / Xj - dpis / Xj)
        term2 = etajs * (dpjs / Xj - dpjs / Xj)
        term3 = tijs * Tijs * (dTijs_Tijs / Xj - dpis / Xj)
        
        total_change = np.sum(term1 + term2 + term3)
        terms.append((term1, term2, term3, total_change))
    
    return terms

# Example data (replace these with actual values)
welfare_change = calculate_welfare_change(trade_flows, exports, price_changes, etajs, tariff_rates, trade_flows_changes)

# Display the results
for industry, result in zip(sigma.keys(), welfare_change):
    term1, term2, term3, total_change = result
    print(f'Industry: {industry}')
    print(f'Term 1: {term1}')
    print(f'Term 2: {term2}')
    print(f'Term 3: {term3}')
    print(f'Total change: {total_change}\n')

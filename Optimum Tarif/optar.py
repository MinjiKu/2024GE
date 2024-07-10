import numpy as np
from scipy.optimize import minimize

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

# lambda = political economy weights
pol_econ = {}

# X_js = the nominal income in industry s of country j
x = {

}
# P_j = the ideal price index in country j
p = {}

class OptimumTariffModel:
    def __init__(self, countries, industries, elasticities, trade_flows, factual_tariffs):
        self.countries = countries
        self.industries = industries
        self.elasticities = elasticities
        self.trade_flows = trade_flows
        self.factual_tariffs = factual_tariffs

    def government_objective(self, tariff_hats, i):
        G_i = 0
        for s in range(len(self.industries)):
            W_is = self.industry_welfare(tariff_hats, i, s)
            G_i += (W_is * pol_econ[i][s])
        return G_i

    def industry_welfare(self, tariff_hats, i, s):
        W_is = 0
        for j in range(len(self.countries)):
            if j != i:
                # W_is += ((tariff_hats[i][j][s] * self.factual_tariffs[i][j][s] - 1) * 
                #          self.trade_flows[j][i][s] * (1 - 1/self.elasticities[s]))
                W_is += (x[j][s] / p[j])
# tariff_hats: the tariff changes for all country pairs and industries
# The term (tariff_hats[i][j][s] * self.factual_tariffs[i][j][s] - 1) represents the net effect of the new tariff on trade costs.
# The term self.trade_flows[j][i][s] * (1 - 1/self.elasticities[s]) adjusts this effect based on the trade flow volume and the industry elasticity.
        return W_is

    # def eq_10(self, tariff_hats, profit_hats, i, s):
    #     # return profit_hats[i][s] - sum(
    #     #     self.trade_flows[i][j][s] * tariff_hats[j][i][s]**(1-self.elasticities[s])
    #     #     for j in range(len(self.countries)) if j != i
    #     # )
    #     sum = 0
    #     for j in range(len(self.countries)):
    #         if j != i:
    #             sum += sum + alpha * (tau)^(-sigma) * (w_hat_new[i])^(1 - sigma) * (P_hat_new[j][s])^(sigma - 1) * X_hat_new[j]
                

    def eq_11(self, profit_hats, wage_hat, i):
        return wage_hat[i] - sum(
            profit_hats[i][s] * (self.elasticities[s] - 1) / self.elasticities[s]
            for s in range(len(self.industries))
        )

    def eq_12(self, tariff_hats, wage_hat, price_index_hats, i, s):
        return price_index_hats[i][s]**(-1) - sum(
            self.trade_flows[j][i][s] * (wage_hat[j] * tariff_hats[i][j][s])**(1-self.elasticities[s])
            for j in range(len(self.countries))
        )

    def eq_13(self, tariff_hats, wage_hat, expenditure_hat, i):
        return (expenditure_hat[i] * sum(sum(self.trade_flows[j][i][s] for s in range(len(self.industries))) for j in range(len(self.countries))) -
                wage_hat[i] * sum(sum(self.trade_flows[i][j][s] for s in range(len(self.industries))) for j in range(len(self.countries))) -
                sum(sum((tariff_hats[i][j][s] * self.factual_tariffs[i][j][s] - 1) * self.trade_flows[j][i][s]
                    for s in range(len(self.industries))) for j in range(len(self.countries)) if j != i))
        
    def optimize_tariffs(self, country_index):
        num_countries = len(self.countries)
        num_industries = len(self.industries)

        # 이곳 채워야 함f
        initial_tariff_hats = np.ones((num_countries, num_countries, num_industries))
        initial_wage_hat = np.ones(num_countries)
        initial_profit_hats = np.ones((num_countries, num_industries))
        initial_price_index_hats = np.ones((num_countries, num_industries))
        initial_expenditure_hat = np.ones(num_countries)

        initial_guess = np.concatenate([
            initial_tariff_hats.flatten(),
            initial_wage_hat,
            initial_profit_hats.flatten(),
            initial_price_index_hats.flatten(),
            initial_expenditure_hat
        ])

        def objective(x):
            tariff_hats = x[:num_countries**2 * num_industries].reshape((num_countries, num_countries, num_industries))
            return -self.government_objective(tariff_hats, country_index)

        def constraints(x):
            tariff_hats = x[:num_countries**2 * num_industries].reshape((num_countries, num_countries, num_industries))
            wage_hat = x[num_countries**2 * num_industries : num_countries**2 * num_industries + num_countries]
            profit_hats = x[num_countries**2 * num_industries + num_countries : num_countries**2 * num_industries + num_countries + num_countries * num_industries].reshape((num_countries, num_industries))
            price_index_hats = x[num_countries**2 * num_industries + num_countries + num_countries * num_industries : -num_countries].reshape((num_countries, num_industries))
            expenditure_hat = x[-num_countries:]

            cons = []
            for i in range(num_countries):
                for s in range(num_industries):
                    cons.append(self.eq_10(tariff_hats, profit_hats, i, s))
                cons.append(self.eq_11(profit_hats, wage_hat, i))
                for s in range(num_industries):
                    cons.append(self.eq_12(tariff_hats, wage_hat, price_index_hats, i, s))
                cons.append(self.eq_13(tariff_hats, wage_hat, expenditure_hat, i))
            
            return cons

        cons = {'type': 'eq', 'fun': constraints}
        bounds = [(0, None)] * (num_countries**2 * num_industries) + [(None, None)] * (len(initial_guess) - num_countries**2 * num_industries)

        result = minimize(objective, initial_guess, method='SLSQP', constraints=cons, bounds=bounds)

        optimal_tariff_hats = result.x[:num_countries**2 * num_industries].reshape((num_countries, num_countries, num_industries))
        optimal_wage_hat = result.x[num_countries**2 * num_industries : num_countries**2 * num_industries + num_countries]

        return optimal_tariff_hats, optimal_wage_hat

# 데이터 설정
countries = ['Korea', 'Germany', 'US', 'China', 'Japan']
industries = ['Gim', 'Car', 'Steel', 'Semi']

# 데이터 
elasticities = np.array([3.5, 2.8, 3.2, 4.0])
trade_flows = np.random.rand(len(countries), len(countries), len(industries)) * 1000
factual_tariffs = np.random.uniform(1, 1.5, (len(countries), len(countries), len(industries)))

# 모델 초기화
model = OptimumTariffModel(countries, industries, elasticities, trade_flows, factual_tariffs)

# 각 국가에 대해 최적 관세 계산
for i, country in enumerate(countries):
    optimal_tariff_hats, optimal_wage_hat = model.optimize_tariffs(i)
    print(f"{country} optimal tariff changes (hats):")
    for j, target_country in enumerate(countries):
        if i != j:
            print(f"  To {target_country}:")
            for k, industry in enumerate(industries):
                print(f"    {industry}: {optimal_tariff_hats[i][j][k]:.2f}")
    print(f"Optimal wage change (hat): {optimal_wage_hat[i]:.2f}")
    print(f"Average optimal tariff change: {np.mean(optimal_tariff_hats[i]):.2f}")
    print()
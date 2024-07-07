import numpy as np
from scipy.optimize import minimize

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
            G_i += W_is
        return G_i

    def industry_welfare(self, tariff_hats, i, s):
        W_is = 0
        for j in range(len(self.countries)):
            if j != i:
                W_is += ((tariff_hats[i][j][s] * self.factual_tariffs[i][j][s] - 1) * 
                         self.trade_flows[j][i][s] * (1 - 1/self.elasticities[s]))
        return W_is

    def constraint_10(self, tariff_hats, profit_hats, i, s):
        return profit_hats[i][s] - sum(
            self.trade_flows[i][j][s] * tariff_hats[j][i][s]**(1-self.elasticities[s])
            for j in range(len(self.countries)) if j != i
        )

    def constraint_11(self, profit_hats, wage_hat, i):
        return wage_hat[i] - sum(
            profit_hats[i][s] * (self.elasticities[s] - 1) / self.elasticities[s]
            for s in range(len(self.industries))
        )

    def constraint_12(self, tariff_hats, wage_hat, price_index_hats, i, s):
        return price_index_hats[i][s]**(-1) - sum(
            self.trade_flows[j][i][s] * (wage_hat[j] * tariff_hats[i][j][s])**(1-self.elasticities[s])
            for j in range(len(self.countries))
        )

    def constraint_13(self, tariff_hats, wage_hat, expenditure_hat, i):
        return (expenditure_hat[i] * sum(sum(self.trade_flows[j][i][s] for s in range(len(self.industries))) for j in range(len(self.countries))) -
                wage_hat[i] * sum(sum(self.trade_flows[i][j][s] for s in range(len(self.industries))) for j in range(len(self.countries))) -
                sum(sum((tariff_hats[i][j][s] * self.factual_tariffs[i][j][s] - 1) * self.trade_flows[j][i][s]
                    for s in range(len(self.industries))) for j in range(len(self.countries)) if j != i))

    def optimize_tariffs(self, country_index):
        num_countries = len(self.countries)
        num_industries = len(self.industries)

        # 이곳 채워야 함
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
                    cons.append(self.constraint_10(tariff_hats, profit_hats, i, s))
                cons.append(self.constraint_11(profit_hats, wage_hat, i))
                for s in range(num_industries):
                    cons.append(self.constraint_12(tariff_hats, wage_hat, price_index_hats, i, s))
                cons.append(self.constraint_13(tariff_hats, wage_hat, expenditure_hat, i))
            
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
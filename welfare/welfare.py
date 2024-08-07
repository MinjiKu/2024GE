import var

def welfare_change(T, X, delta_p, p, pi, t, delta_pi, delta_T):
    delta_W_W = {}
    
    for j in var.countries:  # j국 (수입국)
        term1 = 0
        term2 = 0
        term3 = 0
        
        for i in var.countries:  # i국 (수출국)
            if i != j:
                for s in var.industries:  # s산업
                    term1 += (T[i][j][s] / X[j]) * ((delta_p[j][s] / p[j][s]) - (delta_p[i][s] / p[i][s]))
                    term3 += (t[i][j][s] * T[i][j][s] / X[j]) * ((delta_T[i][j][s] / T[i][j][s]) - (delta_p[i][s] / p[i][s]))
        
        for s in var.industries:  # s산업
            term2 += (pi[j][s] / X[j]) * ((delta_pi[j][s] / pi[j][s]) - (delta_p[j][s] / p[j][s]))
        
        delta_W_W[j] = term1 + term2 + term3
    
    return delta_W_W

print(welfare_change())
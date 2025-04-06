import numpy as np
import math

# paramètres
S0 = 100
K = 100
sigma = 0.2
T = 1.0
valeurs_alpha = [0.97, 0.99, 0.9999]
I0 = 10
alpha = -10
beta = -5
Nmc = 10000

# fonction de répartition de la loi normale
def repartition_normale(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

# prix d'un call européen (r=0)
def call(S, K, sigma, T):
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * repartition_normale(d1) - K * repartition_normale(d2)

# prix d'un put européen (r=0)
def put(S, K, sigma, T):
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * repartition_normale(-d2) - S * repartition_normale(-d1)

# simule une valeur finale S_T
def simuler_ST(S0, sigma, T):
    return S0 * math.exp(-0.5 * sigma**2 * T + sigma * np.random.normal(0, math.sqrt(T)))

# fonction indicatrice pour robbins-monro
def Psi(z, x):
    return 1 if x <= z else 0

# algorithme robbins-monro pour approximer la var
def robbins_monro(pertes, alpha, eta=0.01, n_iter=1000):
    z = 0
    for i in range(n_iter):
        x = np.random.choice(pertes)
        z = z + eta * (alpha - Psi(z, x))
    return z

# calcule var et cvar par méthode de tri
def var_cvar(pertes, alpha):
    pertes = np.sort(pertes)
    k = int(len(pertes) * alpha)
    var = pertes[k]
    cvar = np.mean(pertes[:k+1])
    return var, cvar

# simulation des pertes
#np.random.seed(0)
pertes_call = []
pertes_port = []

V0_call = call(S0, K, sigma, T)
V0_port = I0 * (alpha * V0_call + beta * put(S0, K, sigma, T))

for _ in range(Nmc):
    ST = simuler_ST(S0, sigma, T)
    pertes_call.append(V0_call - max(ST - K, 0))

    Vt = 0
    for _ in range(I0):
        ST_i = simuler_ST(S0, sigma, T)
        Vt += alpha * call(ST_i, K, sigma, T) + beta * put(ST_i, K, sigma, T)
    pertes_port.append(V0_port - Vt)

pertes_call = np.array(pertes_call)
pertes_port = np.array(pertes_port)

# affichage des résultats
for alpha_ in valeurs_alpha:
    v_call, c_call = var_cvar(pertes_call, alpha_)
    v_port, c_port = var_cvar(pertes_port, alpha_)
    rm_call = robbins_monro(pertes_call, alpha_)
    rm_port = robbins_monro(pertes_port, alpha_)

    print(f"α={alpha_}")
    print(f"  var_call_tri={v_call:.2f}")
    print(f"  cvar_call={c_call:.2f}")
    print(f"  var_call_rm={rm_call:.2f}")
    print(f"  var_port_tri={v_port:.2f}")
    print(f"  cvar_port={c_port:.2f}")
    print(f"  var_port_rm={rm_port:.2f}")
    print()

import numpy as np
import math

from matplotlib import pyplot as plt

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

# affichage des résultats pour chaque niveau de confiance alpha_
for alpha_ in valeurs_alpha:
    # calcul de la var et cvar par méthode de tri (ordonnancement) pour le call seul
    v_call, c_call = var_cvar(pertes_call, alpha_)
    # calcul de la var et cvar pour le portefeuille complet (call + put)
    v_port, c_port = var_cvar(pertes_port, alpha_)
    # estimation de la var du call seul par la méthode robbins-monro
    rm_call = robbins_monro(pertes_call, alpha_)
    # estimation de la var du portefeuille par robbins-monro (résultat souvent imprécis)
    rm_port = robbins_monro(pertes_port, alpha_)

    # affichage structuré des résultats
    print(f"α={alpha_}")
    print(f"  var_call_tri={v_call:.2f}")   # var du call (tri)
    print(f"  cvar_call={c_call:.2f}")      # cvar du call (tri)
    print(f"  var_call_rm={rm_call:.2f}")   # var du call (robbins-monro)
    print(f"  var_port_tri={v_port:.2f}")   # var du portefeuille (tri)
    print(f"  cvar_port={c_port:.2f}")      # cvar du portefeuille (tri)
    print(f"  var_port_rm={rm_port:.2f}")   # var du portefeuille (robbins-monro)
    print()



# question 2 : tracer la distribution conditionnelle des pertes > VaR99%
alpha_c = 0.99
var_cond = np.percentile(pertes_port, 100 * alpha_c)
pertes_extremes = pertes_port[pertes_port > var_cond]

# tracer l'histogramme des pertes extrêmes
plt.figure(figsize=(7,4))
plt.hist(pertes_extremes, density = 'true', bins=30, color='darkred', alpha=0.7, edgecolor='black')
plt.title("distribution conditionnelle : pertes > VaR99%")
plt.xlabel("Perte")
plt.ylabel("Fréquence")
plt.grid(True)
plt.tight_layout()
plt.show()

# question 3 : etude de l'influence de la composition du portefeuille
def composition_short(i):
    return -10, -5

def composition_long(i):
    return 10, 5

def composition_mixte(i):
    return (10, 5) if i < I0 // 2 else (-10, -5)

compositions = {
    "Short": composition_short,
    "Long": composition_long,
    "Mixte": composition_mixte
}

resultats = {}

for nom, func in compositions.items():
    pertes = []
    V0 = 0
    for i in range(I0):
        alpha_i, beta_i = func(i)
        V0 += alpha_i * call(S0, K, sigma, T) + beta_i * put(S0, K, sigma, T)

    for j in range(Nmc):
        Vt = 0
        for i in range(I0):
            alpha_i, beta_i = func(i)
            S_Ti = simuler_ST(S0, sigma, T)
            Vt += alpha_i * call(S_Ti, K, sigma, T) + beta_i * put(S_Ti, K, sigma, T)
        pertes.append(V0 - Vt)

    pertes = np.array(pertes)
    var99, cvar99 = var_cvar(pertes, 0.99)
    resultats[nom] = (var99, cvar99)

    # tracer la distribution avec la VaR à 99%
    plt.figure(figsize=(6, 4))
    plt.hist(pertes, density = 'true', bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(var99, color='red', linestyle='--', label=f"VaR 99% = {var99:.2f}")
    plt.title(f"Densité des pertes - {nom}")
    plt.xlabel("Perte")
    plt.ylabel("Densité")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
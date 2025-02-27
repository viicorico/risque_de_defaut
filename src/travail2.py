import numpy as np
import matplotlib.pyplot as plt
import math

T = 1  # 1 an
N = 125
S0 = 100
sigma = 0.4

def generer_mouvement_brownien():
    W = [0]
    delta_t = T / N
    for i in range(N):
        W_i = W[i] + np.sqrt(delta_t) * np.random.normal(0, 1)
        W.append(W_i)
    return W

def simuler_S():
    delta_t = T / N
    t = []
    for i in range(N + 1):
        t.append(i * delta_t)

    W = generer_mouvement_brownien()
    S = []
    for i in range(N + 1):
        S_i = S0 * math.exp(-0.5 * sigma ** 2 * t[i] + sigma * W[i])
        S.append(S_i)
    return t,S

def simuler_S_Nmc(Nmc):
    plt.figure(figsize=(10, 5))

    for _ in range(Nmc):
        t, S = simuler_S()
        plt.plot(t, S, alpha=0.5)

    plt.xlabel("Temps (t)")
    plt.ylabel("S(t)")
    plt.title(f"Simulation de {Nmc} trajectoires de S(t)")
    plt.show()

simuler_S_Nmc(10000)

# Calculer X = S_T - B pour Nmc simulations
def tab_X(Nmc, B):
    X = []
    for i in range(Nmc):
        S_T = simuler_S()[1][-1]  # On récupère S_T (la derniere valeur de S)
        X.append(S_T - B)
    return X

def F(X, a, b, Nx, Nmc):
    x = []
    proba = []

    for i in range(Nx):
        x.append(a + (b - a) * i / Nx)  # Génération des valeurs de x
        compteur = 0

        for j in range(Nmc):
            if X[j] <= x[i]:
                compteur += 1  # On compte les X[j] ≤ x[i]

        proba.append(compteur / Nmc)  # Calcul de la probabilité empirique

    return x, proba

# Paramètres de la simulation
Nmc = 10000
B = 50  # Seuil
X = tab_X(Nmc, B)

# Définir les bornes a et b
a, b = min(X), max(X)

# Calcul de la fonction de répartition empirique
Nx = 100
x, proba = F(X, a, b, Nx, Nmc)

# Affichage
plt.plot(x, proba)
plt.xlabel("x")
plt.ylabel("F_X(x) = P(X ≤ x)")
plt.title(f"Fonction de répartition de X = S_T avec le seuil B= {B}")
plt.grid()
plt.show()
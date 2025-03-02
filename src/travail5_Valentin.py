import numpy as np
import matplotlib.pyplot as plt
import math

# Paramètres globaux
S0 = 100  # Prix initial de l'actif
sigma = 0.4  # Volatilité
T = 1  # Horizon de temps (1 an)
Nmc = 100000  # Nombre de simulations Monte-Carlo
B = 50  # Seuil
alpha_values = [0.1, 0.01]  # Niveaux de confiance (10% et 1%)

# Fonction pour simuler S_T
def simuler_S_T():
    W_T = np.random.normal(0, np.sqrt(T))  # Mouvement brownien à l'horizon T
    S_T = S0 * np.exp((-0.5 * sigma**2) * T + sigma * W_T)
    return S_T

# Fonction pour simuler un échantillon de X = S_T - B
def simuler_echantillon_X(Nmc, B):
    X = []
    for _ in range(Nmc):
        S_T = simuler_S_T()
        X.append(S_T - B)
    return X

# Fonction pour calculer la VaR par ordonnancement
def calculer_var(X, alpha):
    X_ordonne = np.sort(X)  # Ordonner l'échantillon
    k = int(Nmc * alpha)  # Indice du quantile
    VaR = X_ordonne[k]  # VaR = X_{(k)}
    return VaR

# Fonction pour calculer la densité empirique
def f(X, a, b, Nx, Nmc):
    proba = []
    x = []
    for i in range(Nx):
        x.append(a + (b - a) * i / Nx)
        compteur = 0
        for j in range(Nmc):
            if X[j] <= x[i] + (b - a) / Nx and x[i] < X[j]:
                compteur += 1
        proba.append(compteur / (((b - a) / Nx) * Nmc))
    return x, proba

# Simulation de l'échantillon
X = simuler_echantillon_X(Nmc, B)

# Calcul de la VaR pour chaque alpha
for alpha in alpha_values:
    VaR = calculer_var(X, alpha)
    print(f"VaR pour B={B}, alpha={alpha * 100}% : {VaR:.4f}")

# Calcul de la densité empirique
a = min(X)  # Borne inférieure
b = max(X)  # Borne supérieure
Nx = 100  # Nombre de points pour la densité
xdensite, ydensite = f(X, a, b, Nx, Nmc)

# Tracé de la densité empirique de X
plt.figure(figsize=(10, 6))
plt.plot(xdensite, ydensite, label="Densité empirique de X")

# Ajout des lignes verticales pour les VaR
for alpha in alpha_values:
    VaR = calculer_var(X, alpha)
    plt.axvline(x=VaR, color='r' if alpha == 0.1 else 'b', linestyle='--', label=f"VaR (alpha={alpha * 100}%) = {VaR:.2f}")

plt.xlabel("X = S_T - B")
plt.ylabel("Densité")
plt.title(f"Densité empirique de X pour B={B} (Nmc={Nmc})")
plt.legend()
plt.grid()
plt.show()
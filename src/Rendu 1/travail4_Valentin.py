import numpy as np
import matplotlib.pyplot as plt

# Paramètres globaux
S0 = 100  # Prix initial de l'actif
sigma = 0.4  # Volatilité
r = 0  # Taux d'intérêt (supposé non nul pour plus de réalisme)
Nmc = 1000000  # Nombre de simulations Monte-Carlo

# Fonction pour simuler S_T
def simuler_S_T(T):
    Y = np.random.normal(0, 1)  # Loi normale centrée réduite
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Y)
    return S_T

# Fonction indicatrice Ψ(z, x)
def psi(z, x):
    return 1 if x <= z else 0

# Algorithme de Robbins-Monro pour estimer la VaR
def robbins_monro_var(Nmc, alpha, B, T, beta, z0):
    Z = [z0]  # Initialisation de la suite Z
    gamma = [beta / ((n + 1) ** 0.9) for n in range(Nmc)]  # Suite des pas gamma_n

    for n in range(Nmc):
        # Simuler S_T et calculer X_n = S_T - B
        S_T = simuler_S_T(T)
        X_n = S_T - B

        # Fonction indicatrice Psi(Z_n, X_n)
        psi_n = psi(Z[n], X_n)

        # Mise à jour de Z_n
        Z_next = Z[n] - gamma[n] * (psi_n - alpha)
        Z.append(Z_next)

    return Z

# Paramètres à tester
parametres = [
    {"B": 100, "alpha": 0.01, "T": 1, "beta": 10, "z0": 0},
    {"B": 100, "alpha": 0.001, "T": 1, "beta": 10, "z0": 0},
    {"B": 100, "alpha": 0.01, "T": 10/365, "beta": 10, "z0": 0},
    {"B": 100, "alpha": 0.001, "T": 10/365, "beta": 10, "z0": 0},
    {"B": 50, "alpha": 0.01, "T": 1, "beta": 10, "z0": 0},
    {"B": 50, "alpha": 0.001, "T": 1, "beta": 10, "z0": 0},
    {"B": 36, "alpha": 0.01, "T": 1, "beta": 10, "z0": 0},
]

# Boucle sur les paramètres pour générer les graphiques
for params in parametres:
    B = params["B"]
    alpha = params["alpha"]
    T = params["T"]
    beta = params["beta"]
    z0 = params["z0"]

    # Exécution de l'algorithme
    Z = robbins_monro_var(Nmc, alpha, B, T, beta, z0)

    # Tracé de la convergence
    plt.figure(figsize=(10, 6))
    plt.plot(Z, label=f"B={B}, alpha={alpha}, T={T}, beta={beta}, z0={z0}")
    plt.axhline(y=Z[-1], color='r', linestyle='--', label=f"VaR estimée = {Z[-1]:.2f}")
    plt.xlabel("Itérations")
    plt.ylabel("Z_n")
    plt.title(f"Convergence de l'algorithme de Robbins-Monro pour B={B}, alpha={alpha}, T={T}")
    plt.legend()
    plt.grid()
    plt.show()

    # Affichage de la VaR estimée
    print(f"VaR pour B={B}, alpha={alpha}, T={T} : {Z[-1]:.4f}")

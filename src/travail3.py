import numpy as np
import matplotlib.pyplot as plt
import math

# ====================================================
# Paramètres globaux
# ====================================================
S0 = 100  # Prix initial de l'actif
sigma = 0.4  # Volatilité
T = 1  # Horizon de temps
N = 100  # Nombre de pas de temps
dt = T / N  # Pas de temps
B = 100  # Seuil pour la perte X = S_T - B
Nmc = 10000  # Nombre de simulations Monte-Carlo
alpha = 0.5  # Niveau de confiance (α = 1/2 pour la loi normale)

# ====================================================
# Fonctions pour la simulation de S(t)
# ====================================================

def generer_mouvement_brownien():
    """Génère un mouvement brownien standard."""
    W = [0]
    for _ in range(N):
        W_i = W[-1] + np.sqrt(dt) * np.random.normal(0, 1)
        W.append(W_i)
    return W

def simuler_S():
    """Simule une trajectoire de S(t) selon le modèle géométrique brownien."""
    t = np.linspace(0, T, N + 1)
    W = generer_mouvement_brownien()
    S = [S0 * math.exp(-0.5 * sigma ** 2 * t[i] + sigma * W[i]) for i in range(N + 1)]
    return t, S

# ====================================================
# Algorithme de Robbins-Monro pour α = 1/2
# ====================================================

def robbins_monro_normal(Nmc, beta, z0):
    """
    Implémente l'algorithme de Robbins-Monro pour trouver z* tel que F(z*) = α = 1/2.
    """
    Z = [z0]  # Initialisation de la suite Z
    gamma = [beta / ((n + 1) ** 0.9) for n in range(Nmc)]  # Suite des pas gamma_n

    for n in range(Nmc):
        # Simuler S_T et calculer X_n = S_T - B
        S_T = simuler_S()[1][-1]
        X_n = S_T - B

        # Fonction indicatrice Psi(Z_n, X_n)
        psi = 1 if X_n <= Z[n] else 0

        # Mise à jour de Z_n
        Z_next = Z[n] - gamma[n] * (psi - alpha)#on peut mettre alpha en paramètres si besoin
        Z.append(Z_next)

    return Z

# ====================================================
# Génération des graphiques pour chaque combinaison de paramètres
# ====================================================

# Paramètres à tester
parametres = [
    {"z0": 1, "beta": 10},
    {"z0": 1, "beta": 1},
    {"z0": 0.1, "beta": 1},
    {"z0": 1, "beta": 0.1},
    {"z0": 1, "beta": 100},
    {"z0": 1, "beta": 1000},
]

# Boucle sur les paramètres pour générer les graphiques
for params in parametres:
    z0 = params["z0"]
    beta = params["beta"]

    # Exécution de l'algorithme
    Z = robbins_monro_normal(Nmc, beta, z0)

    # Tracé de la convergence
    plt.figure(figsize=(10, 6))
    plt.plot(Z, label=f"z0={z0}, beta={beta}")
    plt.axhline(y=0, color='r', linestyle='--', label="z* = 0")
    plt.xlabel("Itérations")
    plt.ylabel("Z_n")
    plt.title(f"Convergence de l'algorithme de Robbins-Monro pour α = 1/2\nz0={z0}, beta={beta}")
    plt.legend()
    plt.grid()
    plt.show()
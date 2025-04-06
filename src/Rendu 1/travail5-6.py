import numpy as np
import matplotlib.pyplot as plt
import math

# ====================================================
# Fonctions pour la densité Beta (sans scipy)
# ====================================================

def gamma_function(z):
    """
    Calcule la fonction Gamma pour z > 0 en utilisant l'approximation de Stirling.
    """
    if z == 1:
        return 1
    elif z == 0.5:
        return math.sqrt(math.pi)
    else:
        return math.sqrt(2 * math.pi / z) * ((z / math.e) ** z)

def beta_function(alpha, beta):
    """
    Calcule la fonction Beta B(alpha, beta) en utilisant la fonction Gamma.
    """
    return gamma_function(alpha) * gamma_function(beta) / gamma_function(alpha + beta)

def beta_pdf(x, alpha, beta):
    """
    Calcule la densité de la loi Beta en x pour les paramètres alpha et beta.
    """
    if x < 0 or x > 1:
        return 0  # La densité est nulle en dehors de [0, 1]
    else:
        return (x ** (alpha - 1)) * ((1 - x) ** (beta - 1)) / beta_function(alpha, beta)

# ====================================================
# Simulation de la loi Beta par la méthode de rejet
# ====================================================

def simulate_beta_rejection(alpha, beta, Nmc):
    """
    Simule des variables aléatoires suivant une loi Beta(alpha, beta) par la méthode de rejet.
    """
    Y = []
    x0 = (alpha - 1) / (alpha + beta - 2)  # Point où la densité atteint son maximum
    C = beta_pdf(x0, alpha, beta)  # Constante de rejet

    for _ in range(Nmc):
        while True:
            U = np.random.uniform(0, 1)  # Simuler U ~ U(0, 1)
            Y_candidate = np.random.uniform(0, 1)  # Simuler Y ~ U(0, 1)
            f = beta_pdf(Y_candidate, alpha, beta)  # Densité Beta
            g = 1  # Densité de la loi uniforme U(0, 1)

            if U <= f / (C * g):
                Y.append(Y_candidate)
                break
    return Y

# ====================================================
# Fonctions pour afficher la fonction de densité empirique
# ====================================================

def f(X, a, b, Nx, Nmc):
    """
    Calcule la fonction de densité empirique.
    """
    proba = []
    x = []
    for i in range(Nx):
        x.append(a + (b - a) * i / Nx)
        compteur = 0
        for j in range(Nmc):
            if X[j] <= x[i] + (b - a) / Nx and x[i] < X[j]:
                compteur += 1
        proba.append(compteur / (((b - a) / Nx) * Nmc))
    return (x, proba)

# ====================================================
# Fonction principale (main)
# ====================================================

def main():
    # Paramètres
    Nmc = 10000  # Nombre de simulations Monte-Carlo
    a = 0  # Borne inférieure de l'intervalle
    b = 1  # Borne supérieure de l'intervalle
    Nx = 100  # Nombre de points pour l'affichage de la densité

    # Cas à simuler : (alpha, beta) = (2, 5), (2, 2), (5, 1)
    cases = [(2, 5), (2, 2), (5, 1)]

    # Boucle sur chaque cas
    for alpha, beta in cases:
        # Simuler la loi Beta
        Y = simulate_beta_rejection(alpha, beta, Nmc)

        # Calculer la densité empirique
        x_densite, y_densite = f(Y, a, b, Nx, Nmc)

        # Tracer la densité empirique
        plt.figure(figsize=(8, 6))
        plt.plot(x_densite, y_densite, label=f"Densité empirique Beta({alpha}, {beta})")
        plt.title(f"Densité de la loi Beta({alpha}, {beta})")
        plt.xlabel("x")
        plt.ylabel("Densité")
        plt.legend()
        plt.grid()
        plt.show()

# ====================================================
# Point d'entrée du script
# ====================================================

if __name__ == "__main__":
    main()
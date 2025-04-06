import numpy as np
import matplotlib.pyplot as plt

# ====================================================
# Fonction pour calculer la densité empirique
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
# Simulation 1 : Calcul de P[Y > 5] dans l'espace Q
# ====================================================

def simulation_1(Nmc):
    """
    Simulation classique pour calculer P[Y > 5] dans l'espace Q.
    Y suit une loi N(0, 1).
    """
    # Simuler Y ~ N(0, 1)
    Y = np.random.normal(0, 1, Nmc)

    # Compter le nombre de réalisations où Y > 5
    n5 = np.sum(Y > 5)

    # Calculer P[Y > 5]
    P_Y_gt_5 = n5 / Nmc

    return Y, P_Y_gt_5

# ====================================================
# Simulation 2 : Calcul de EQ[I_{Y_Q > 5}] dans l'espace P
# ====================================================

def simulation_2(Nmc, mu, theta, T):
    """
    Calcul de EQ[I_{Y_Q > 5}] dans l'espace P en utilisant l'échantillonnage d'importance.
    YP suit une loi N(mu, 1).
    """
    # Simuler YP ~ N(mu, 1)
    YP = mu + np.random.normal(0, 1, Nmc)

    # Calculer les poids d'importance : exp(-(gi + mu) * theta - theta^2 * T / 2)
    weights = np.exp(-(YP) * theta - (theta**2 * T) / 2)

    # Compter le nombre de réalisations où YP > 5
    n5 = np.sum(YP > 5)

    # Calculer EQ[I_{Y_Q > 5}] = EP[I_{YP > 5} * poids]
    EQ_I_YQ_gt_5 = np.mean((YP > 5) * weights)

    return YP, EQ_I_YQ_gt_5

# ====================================================
# Fonction principale (main)
# ====================================================

def main():
    # Paramètres
    Nmc_1 = 100000  # Nombre de simulations pour Simulation 1
    Nmc_2 = 100000  # Nombre de simulations pour Simulation 2
    mu = 5         # Moyenne de la loi Q
    theta = 1      # Paramètre theta
    T = 1          # Paramètre T
    a = -10        # Borne inférieure pour l'affichage de la densité
    b = 10         # Borne supérieure pour l'affichage de la densité
    Nx = 100       # Nombre de points pour l'affichage de la densité

    # Simulation 1
    Y_1, P_Y_gt_5 = simulation_1(Nmc_1)
    print(f"Simulation 1 : P[Y > 5] = {P_Y_gt_5}")

    # Simulation 2
    Y_2, EQ_I_YQ_gt_5 = simulation_2(Nmc_2, mu, theta, T)
    print(f"Simulation 2 : EQ[I_{{Y_Q > 5}}] = {EQ_I_YQ_gt_5}")

    # Affichage des densités empiriques
    x_1, y_1 = f(Y_1, a, b, Nx, Nmc_1)
    x_2, y_2 = f(Y_2, a, b, Nx, Nmc_2)

    plt.figure(figsize=(12, 6))

    # Densité de Y dans Simulation 1
    plt.subplot(1, 2, 1)
    plt.plot(x_1, y_1, label="Densité empirique de Y ~ N(0, 1)")
    plt.title("Simulation 1 : Densité de Y")
    plt.xlabel("x")
    plt.ylabel("Densité")
    plt.legend()
    plt.grid()

    # Densité de YP dans Simulation 2
    plt.subplot(1, 2, 2)
    plt.plot(x_2, y_2, label=f"Densité empirique de YP ~ N({mu}, 1)")
    plt.title("Simulation 2 : Densité de YP")
    plt.xlabel("x")
    plt.ylabel("Densité")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# ====================================================
# Point d'entrée du script
# ====================================================

if __name__ == "__main__":
    main()
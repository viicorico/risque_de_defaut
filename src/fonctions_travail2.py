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
    plt.savefig("../images/travail2/plot_simulation_S_t.png")  # Enregistrement en PNG
    plt.show()


# Calculer X = S_T - B pour Nmc simulations
def tab_X(Nmc, B):
    X = []
    for i in range(Nmc):
        S_T = simuler_S()[1][-1]  # On récupère S_T (la derniere valeur de S)
        X.append(S_T - B)
    return X

def fonction_repartition(X, a, b, Nx, Nmc):
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

def densite_empirique(X, a, b, Nx, Nmc):
    proba = []
    x = []
    for i in range(0, Nx):
        x.append(a + (b - a) * i / Nx)
        compteur = 0

        for j in range(1, Nmc):
            if (X[j] <= x[i] + (b - a) / Nx and x[i] < X[j]):
                compteur += 1
        proba.append(compteur / (((b - a) / Nx) * Nmc))
    return (x, proba)


def tracer_fonction_repartition(X, Nx, Nmc, B, save_path=None):
    a, b = min(X), max(X)
    x, proba = fonction_repartition(X, a, b, Nx, Nmc)

    plt.figure(figsize=(8, 5))
    plt.plot(x, proba, label="F_X(x) = P(X ≤ x)", color='blue')
    plt.xlabel("x")
    plt.ylabel("F_X(x)")
    plt.title("Fonction de répartition empirique de X")
    plt.grid()
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def tracer_densite(X, Nx, Nmc, B, save_path=None):
    """ Affiche la densité empirique de X avec une boucle. """
    a, b = min(X), max(X)
    x, proba = densite_empirique(X, a, b, Nx, Nmc)

    plt.figure(figsize=(8, 5))
    plt.plot(x, proba, label="Densité empirique de X", color='red')
    plt.xlabel("x")
    plt.ylabel("f_X(x)")
    plt.title("Densité empirique de X")
    plt.grid()
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

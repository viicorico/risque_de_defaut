import numpy as np
import matplotlib.pyplot as plt


def simuler_trajectoire(S0, r, sigma, T, N, dt):
    """
    Simule une seule trajectoire de l'évolution d'un actif selon un mouvement géométrique brownien.

    Paramètres :
    - S0 : Prix initial de l'actif
    - r : Taux d'intérêt
    - sigma : Volatilité
    - T : Horizon de temps
    - N : Nombre de pas de temps
    - dt : Pas de temps

    Retourne :
    - t : Tableau des instants de temps
    - S : Tableau des prix simulés de l'actif
    """
    t = np.linspace(0, T, N + 1)  # Instants de temps
    S = np.zeros(N + 1)  # Stockage des valeurs de l'actif
    S[0] = S0  # Condition initiale

    for i in range(1, N + 1):
        Z = np.random.normal()  # Bruit gaussien standard
        S[i] = S[i - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    return t, S


def afficheTrajectoires(S0, r, sigma, T, N, Nmc, dt, B):
    """
    Simule et trace Nmc trajectoires de l'évolution d'un actif selon un mouvement géométrique brownien.
    Colore en rouge les trajectoires pour lesquelles S_T < B.

    Paramètres :
    - S0 : Prix initial de l'actif
    - r : Taux d'intérêt
    - sigma : Volatilité
    - T : Horizon de temps
    - N : Nombre de pas de temps
    - Nmc : Nombre de simulations Monte-Carlo
    - dt : Pas de temps
    - B : Seuil à comparer avec S_T

    Retourne :
    - None (Affiche un graphe avec les trajectoires simulées et affiche la probabilité P(S_T < B))
    """
    t = np.linspace(0, T, N + 1)  # Instants de temps
    plt.figure(figsize=(10, 6))  # Taille du graphe

    count_ST_inferieur_B = 0  # Compteur du nombre de trajectoires où S_T < B

    # Génération de Nmc trajectoires
    for _ in range(Nmc):
        _, S = simuler_trajectoire(S0, r, sigma, T, N, dt)  # Simulation de la trajectoire

        if S[-1] < B:
            plt.plot(t, S, color='red', alpha=0.7)  # Rouge si S_T < B
            count_ST_inferieur_B += 1  # Incrémenter le compteur
        else:
            plt.plot(t, S, color='blue', alpha=0.7)  # Bleu sinon

    # Calcul de la probabilité estimée P(S_T < B)
    proba = count_ST_inferieur_B / Nmc

    # Paramètres du graphe
    plt.xlabel("Temps")
    plt.ylabel("Prix de l'actif")
    plt.title(f"Simulation de {Nmc} trajectoires d'évolution de l'actif\n"
              f"Trajectoires en rouge si S_T < {B} (Probabilité estimée : {proba:.4f})")
    plt.grid(True)
    plt.show()

    # Affichage de la probabilité
    print(f"Probabilité estimée P(S_T < {B}) = {proba:.4f}")


if __name__ == '__main__':
    Nmc = 100
    S0 = 100
    r = 0
    sigma = 0.4
    T = 1
    N = 100
    B = 100  # Seuil pour la condition S_T < B
    dt = T / N

    afficheTrajectoires(S0, r, sigma, T, N, Nmc, dt, B)
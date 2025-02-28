from matplotlib import pyplot as plt

from fonctions_travail2 import tab_X, tracer_fonction_repartition, tracer_densite

# Paramètres globaux
T = 1        # Durée (1 an)
N = 125      # Nombre de pas de temps
S0 = 100     # Prix initial
sigma = 0.4  # Volatilité
Nmc = 10000  # Nombre de simulations
Nx = 100     # Nombre de points pour la répartition

if __name__ == "__main__":
    print("Simulation en cours...")

    # Simulation et génération de X = S_T - B
    for B in (36,50,100):
        X = tab_X(Nmc, B)

        # Tracer la fonction de répartition et la densité empirique
        tracer_fonction_repartition(X, Nx, Nmc, B, save_path=f"../images/travail2/plot_fonction_repartition_B_{B}.png")
        tracer_densite(X, Nx, Nmc, B, save_path=f"../images/travail2/plot_densite_B_{B}.png")

        print("Terminé ! Les graphiques ont été sauvegardés.")

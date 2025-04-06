import numpy as np
import matplotlib.pyplot as plt

# Paramètres du modèle
N = 125               # Nombre d'entreprises
T = 1.0               # Durée de l'horizon (en années)
n_steps = 12          # Observation mensuelle → 13 dates : t_0, ..., t_12
dt = T / n_steps      # Pas de temps
S0 = 100              # Valeur initiale de chaque actif
B = 50                # Seuil de défaut
sigma = 0.4           # Volatilité constante
R = 0.3               # Taux de recouvrement constant
Nmc = 1000            # Nombre de simulations Monte Carlo (à augmenter si besoin)

# Liste des temps de simulation
times = [k * dt for k in range(n_steps + 1)]
#Ici on initialise tout ce qui est fixé dans l'énoncé. On va simuler la trajectoire de chaque entreprise à ces dates mensuelles.

# Résultats à stocker
L_star_list = []      # Nombre de défauts par simulation
Pi_star_list = []     # Dette totale Π_T^* par simulation


# Simulation des trajectoires et détection du défaut
for sim in range(Nmc):
    L_star = 0        # Nombre de défauts dans ce scénario
    Pi_star = 0.0     # Dette dans ce scénario

    for i in range(N):  # Pour chaque entreprise
        S = S0
        defaulted = False

        for t in times:  # Pour chaque date mensuelle
            Z = np.random.randn()  # Incrément brownien
            S = S * np.exp(-0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * Z)

            # Si l'entreprise passe sous le seuil B pour la première fois
            if not defaulted and S <= B:
                L_star += 1
                Pi_star += R * S  # On comptabilise la dette à cette date
                defaulted = True  # On ne compte qu'une fois le défaut

    # On stocke le résultat de cette simulation
    L_star_list.append(L_star)
    Pi_star_list.append(Pi_star)


# Convertir les listes en tableaux numpy
L_star_array = np.array(L_star_list)
Pi_star_array = np.array(Pi_star_list)


# Calcul de P[L* ≥ K]
K_vals = list(range(1, 101))  # Valeurs de K de 1 à 100
P_L_geq_K = []

for K in K_vals:
    nb_geq = np.sum(L_star_array >= K)  # Nombre de fois où L* ≥ K
    proba = nb_geq / Nmc
    P_L_geq_K.append(proba)

#Calcul de E[Π_T^* | L* > K]
K_plot = list(range(10, 101, 10))  # Tous les 10 défauts
E_Pi_cond = []

for K in K_plot:
    indices = L_star_array > K
    if np.any(indices):
        expectation = np.mean(Pi_star_array[indices])
    else:
        expectation = 0  # Aucun scénario ne satisfait la condition
    E_Pi_cond.append(expectation)

#Fonction de répartition de la dette Π_T^*
x_vals = np.linspace(0, max(Pi_star_array), 200)
cdf_vals = []

for x in x_vals:
    cdf_vals.append(np.mean(Pi_star_array <= x))


# Graphique 1 : P[L* ≥ K]
plt.figure()
plt.plot(K_vals, P_L_geq_K)
plt.xlabel("K")
plt.ylabel("P[L* ≥ K]")
plt.title("Probabilité que le nombre de défauts dynamiques ≥ K")
plt.grid()

# Graphique 2 : E[Π_T^* | L* > K]
plt.figure()
plt.plot(K_plot, E_Pi_cond, marker='o')
plt.xlabel("K")
plt.ylabel("E[Π_T^* | L* > K]")
plt.title("Espérance de la dette conditionnelle")
plt.grid()

# Graphique 3 : Fonction de répartition de la dette Π_T^*
plt.figure()
plt.plot(x_vals, cdf_vals)
plt.xlabel("x")
plt.ylabel("P[Π_T^* ≤ x]")
plt.title("Fonction de répartition de la dette Π_T^*")
plt.grid()

plt.show()

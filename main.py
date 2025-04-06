import numpy as np
import matplotlib.pyplot as plt

# Paramètres
N = 100  # Nombre d'entreprises
T = 1.0  # Horizon de simulation (1 an)
S0 = 100  # Valeur initiale des actifs des entreprises
mu = 0.0  # Rendement moyen
sigma = 0.2  # Volatilité des actifs
rho = 0.5  # Corrélation entre les actifs (modifié selon les scénarios)
B = 36  # Seuil de défaut
n_simulations = 10000  # Nombre de simulations

taux_recouvrement = 0.3

def decompose_cholesky(cov_matrix):
    return np.linalg.cholesky(cov_matrix)

def simuler_trajectoires_corrigees():
    cov_matrix = rho * np.ones((N, N)) + (1 - rho) * np.eye(N)
    chol_matrix = decompose_cholesky(cov_matrix)
    Z = np.random.normal(0, 1, (n_simulations, N))  # Variables aléatoires normales
    Z_corrigees = Z @ chol_matrix.T  # Applique la matrice de Cholesky pour corréler les actifs
    ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_corrigees)  # Simulation des prix
    return ST

def simuler_independants():
    Z = np.random.normal(0, 1, (n_simulations, N))  # Variables normales indépendantes
    ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)  # Simulation des prix
    return ST

def calculer_distribution_pertes(ST, taux_recouvrement, B):
    defauts = ST < B  
    pertes = (1 - taux_recouvrement) * np.where(defauts, ST, 0)  # Pertes en cas de défaut
    perte_totale=pertes.sum(axis=1)  # Somme des pertes pour chaque simulation
    return perte_totale

def tracer_cdf(pertes_indep, pertes_corr, seuils):
    x = np.linspace(0, 40, 500)  # Plage de valeurs à analyser
    cdf_indep = [np.mean(pertes_indep <= xi) for xi in x]
    cdf_corr = [np.mean(pertes_corr <= xi) for xi in x]

    plt.figure(figsize=(10, 6))
    plt.plot(x, cdf_indep, label='Indépendant (rho=0)', color='blue')
    plt.plot(x, cdf_corr, label=f'Corrélé (rho={rho})', color='red')
    plt.xlabel("Dette totale agrégée (ΠT)")
    plt.ylabel("Fonction de répartition P[ΠT ≤ x]")
    plt.title(f"Fonction de répartition de la dette pour B = {seuils}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Fonction de répartition de la dette")
    plt.show()

def comparer_seuils(seuils):
    for seuil in seuils:
        pertes_indep = calculer_distribution_pertes(simuler_independants(), taux_recouvrement, seuil)
        pertes_corr = calculer_distribution_pertes(simuler_trajectoires_corrigees(), taux_recouvrement, seuil)
        tracer_cdf(pertes_indep, pertes_corr, seuil)

seuils = [100, 50, 36]
comparer_seuils(seuils)


import numpy as np
import matplotlib.pyplot as plt
import math

# ====================================================
# Simulation de S_T et calcul de X = S_T - B
# ====================================================

def simuler_S_T(S0, r, sigma, T):
    """
    Simule la valeur de l'actif à l'horizon T selon le modèle de mouvement brownien géométrique.
    
    Formule : 
        S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Y),
    où Y ~ N(0,1).
    
    Paramètres :
        S0    : float, prix initial.
        r     : float, taux d'intérêt.
        sigma : float, volatilité.
        T     : float, horizon de temps.
        
    Renvoie :
        S_T : float, valeur simulée de l'actif à T.
    """
    Y = np.random.normal(0, 1)
    S_T = S0 * math.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Y)
    return S_T

def simuler_X(S0, r, sigma, T, B):
    """
    Calcule X = S_T - B en simulant S_T.
    
    Paramètres :
        S0, r, sigma, T : paramètres pour simuler S_T.
        B              : float, seuil.
    
    Renvoie :
        X : float, la variable d'intérêt.
    """
    S_T = simuler_S_T(S0, r, sigma, T)
    return S_T - B

# ====================================================
# Fonction indicatrice Ψ(z, x)
# ====================================================

def Psi(z, x):
    """
    Renvoie 1 si x est inférieur ou égal à z, sinon 0.
    
    Paramètres :
        z : float, valeur candidate de VaR.
        x : float, réalisation de X.
    
    Renvoie :
        int : 1 si x <= z, 0 sinon.
    """
    return 1 if x <= z else 0

# ====================================================
# Algorithme de Robbins-Monro pour estimer la VaR
# ====================================================

def robbins_monro_var(S0, r, sigma, T, B, alpha, beta, z0, Nmc, lambda_decay=0.9):
    """
    Implémente l'algorithme de Robbins-Monro pour trouver z* tel que
        P[X <= z*] = alpha,
    où X = S_T - B.
    
    Mise à jour : 
        z_{n+1} = z_n - γ_n (Ψ(z_n, X_n) - alpha)
    avec 
        γ_n = beta / ((n+1)^lambda_decay)
    
    Paramètres :
        S0          : float, prix initial.
        r           : float, taux d'intérêt.
        sigma       : float, volatilité.
        T           : float, horizon de temps.
        B           : float, seuil pour X.
        alpha       : float, niveau de risque (ex. 0.01 pour 1%).
        beta        : float, paramètre du pas d'apprentissage.
        z0          : float, estimation initiale de VaR.
        Nmc         : int, nombre d'itérations.
        lambda_decay: float, exponent pour la décroissance du pas (souvent 0.9).
    
    Renvoie :
        z       : float, estimation finale de z*.
        history : liste, historique des valeurs de z (pour visualiser la convergence).
    """
    z = z0
    history = [z]
    
    for n in range(Nmc):
        gamma_n = beta / ((n + 1) ** lambda_decay)
        x_n = simuler_X(S0, r, sigma, T, B)
        psi_val = Psi(z, x_n)
        z = z - gamma_n * (psi_val - alpha)
        history.append(z)
    
    return z, history

# ====================================================
# Estimation de la VaR par méthode empirique (ordonnancement)
# ====================================================

def empirical_var(S0, r, sigma, T, B, alpha, Nmc):
    """
    Calcule la VaR de façon empirique par simulation Monte Carlo.
    
    Pour un échantillon de Nmc réalisations de X, on estime z* tel que :
        P[X <= z*] = alpha.
    La VaR positive est alors définie par : VaR = -z*.
    
    Paramètres :
        S0, r, sigma, T : paramètres pour simuler S_T.
        B              : float, seuil pour X.
        alpha          : float, niveau de risque (ex. 0.01 pour 1%).
        Nmc            : int, nombre de simulations.
        
    Renvoie :
        VaR_empirique : float, VaR estimée (valeur positive).
        X_vals        : array, échantillon des réalisations de X.
    """
    X_vals = np.array([simuler_X(S0, r, sigma, T, B) for _ in range(Nmc)])
    z_empirical = np.quantile(X_vals, alpha)
    VaR_empirique = -z_empirical if z_empirical < 0 else 0
    return VaR_empirique, X_vals


# ====================================================
# Fonction principale pour exécuter les cas de test
# ====================================================

def main():
    # Paramètres globaux communs
    S0 = 100
    r = 0.0
    sigma = 0.4
    Nmc = 10000      # Nombre d'itérations/simulations
    beta = 1.0       # Paramètre du pas d'apprentissage pour Robbins-Monro
    z0 = 0.0         # Estimation initiale de z
    
    # Liste de cas de test : (B, alpha, T, description)
    cas_test = [
        (100, 0.01, 1.0, "B=100, α=1%, T=1 an"),
        (100, 0.001, 1.0, "B=100, α=0.1%, T=1 an"),
        (100, 0.01, 10/365, "B=100, α=1%, T=10 jours"),
        (100, 0.001, 10/365, "B=100, α=0.1%, T=10 jours"),
        (50, 0.01, 1.0, "B=50, α=1%, T=1 an"),
        (50, 0.001, 1.0, "B=50, α=0.1%, T=1 an"),
        (36, 0.01, 1.0, "B=36, α=1%, T=1 an")
    ]
    
    for B, alpha, T, description in cas_test:
        # Estimation par l'algorithme de Robbins-Monro
        final_z, history = robbins_monro_var(S0, r, sigma, T, B, alpha, beta, z0, Nmc)
        # Comme X = S_T - B, le quantile z* est négatif pour des faibles α
        VaR_RM = -final_z if final_z < 0 else 0

        # Estimation empirique par ordonnancement
        VaR_empirique, X_vals = empirical_var(S0, r, sigma, T, B, alpha, Nmc)
        
        # Affichage des résultats
        print(f"{description} -> VaR (Robbins-Monro) : {VaR_RM:.4f} euros, VaR empirique : {VaR_empirique:.4f} euros")
        


if __name__ == "__main__":
    main()

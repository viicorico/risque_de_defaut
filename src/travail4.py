import numpy as np
import math

# --------------------------------------------------------------------
# 1) Simulation de S_T
# --------------------------------------------------------------------
def simulate_ST(T, S0, r, sigma):
    """
    Simule la valeur finale S_T après un temps T en utilisant le modèle
    du mouvement géométrique brownien.
    
    Formule :
      S_T = S0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * N(0,1))
    """
    return S0 * math.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * np.random.normal())

# --------------------------------------------------------------------
# 2) Algorithme de Robbins–Monro pour la VaR
# --------------------------------------------------------------------
def robbins_monro_VaR(Nmc, beta, z0, alpha, T, B, S0, r, sigma):
    """
    Trouve z* tel que P(S_T - B <= z*) = alpha, via l'algorithme de Robbins–Monro.
    
    Paramètres :
      - Nmc  : Nombre d'itérations (simulations)
      - beta : Paramètre pour le pas (gamma_n = beta / (n+1)^0.9)
      - z0   : Estimation initiale de z
      - alpha: Niveau de risque (ex. 0.01 pour 1%)
      - T    : Horizon (en années)
      - B    : Seuil de comparaison
      - S0, r, sigma : Paramètres pour simuler S_T
      
    Retourne :
      - La dernière valeur z* obtenue (qui est négative en cas de perte)
    """
    Z = [z0]
    for n in range(Nmc):
        S_T = simulate_ST(T, S0, r, sigma)
        X = S_T - B
        psi = 1 if X <= Z[n] else 0
        gamma_n = beta / ((n+1)**0.9)
        Z_next = Z[n] - gamma_n * (psi - alpha)
        Z.append(Z_next)
    return Z[-1]

# --------------------------------------------------------------------
# 3) Calcul de la VaR pour différents scénarios
# --------------------------------------------------------------------
def calculVars():
    """
    Calcule la VaR (en valeur positive) pour différents scénarios et
    renvoie une liste de résultats.
    
    Chaque résultat est un dictionnaire contenant :
      - "B"            : La valeur de B
      - "T"            : L'horizon, en années
      - "alpha"        : Le niveau de risque (en décimal)
      - "VaR_positive" : La VaR sous forme positive (ex. 15.7984)
    """
    # Paramètres globaux
    S0    = 100
    r     = 0
    sigma = 0.4
    Nmc   = 10000
    beta  = 1
    z0    = -10  # estimation initiale pour z

    # Scénarios demandés :
    scenarios = [
        {"B": 100, "T": 1,      "alpha": 0.01,  "label": "B=100, T=1 an, α=1%"},
        {"B": 100, "T": 1,      "alpha": 0.001, "label": "B=100, T=1 an, α=0.1%"},
        {"B": 100, "T": 10/365, "alpha": 0.01,  "label": "B=100, T=10 jours, α=1%"},
        {"B": 100, "T": 10/365, "alpha": 0.001, "label": "B=100, T=10 jours, α=0.1%"},
        {"B": 50,  "T": 1,      "alpha": 0.01,  "label": "B=50, T=1 an, α=1%"},
        {"B": 50,  "T": 1,      "alpha": 0.001, "label": "B=50, T=1 an, α=0.1%"},
        {"B": 36,  "T": 1,      "alpha": 0.01,  "label": "B=36, T=1 an, α=1%"}
    ]

    results = []
    for sc in scenarios:
        B_case     = sc["B"]
        T_case     = sc["T"]
        alpha_case = sc["alpha"]

        z_star = robbins_monro_VaR(Nmc, beta, z0, alpha_case, T_case, B_case, S0, r, sigma)
        VaR_positive = -z_star  # On prend l'opposé pour avoir la perte en valeur positive

        # Stocke également les valeurs de B, T et alpha pour l'affichage
        results.append({
            "B": B_case,
            "T": T_case,
            "alpha": alpha_case,
            "VaR_positive": VaR_positive
        })

    return results

# --------------------------------------------------------------------
# 4) Affichage des résultats
# --------------------------------------------------------------------
def afficherVars(results):
    """
    Affiche les résultats sous la forme :
      B=<valeur>, T=<valeur> an, α=<valeur>%  VaR <valeur>
    """
    for res in results:
        # On formate alpha en pourcentage et T en "an" (si T < 1, on affiche en fraction d'année)
        B = res["B"]
        T = res["T"]
        alpha_pct = res["alpha"] * 100  # convertir en pourcentage
        VaR_positive = res["VaR_positive"]
        print(f"B={B}, T={T} an, α={alpha_pct:.1f}%  VaR {VaR_positive:.4f}")

# --------------------------------------------------------------------
# 5) Point d'entrée du script
# --------------------------------------------------------------------
if __name__ == "__main__":
    resultats = calculVars()  # Calcul et stockage des résultats
    afficherVars(resultats)   # Affichage des résultats

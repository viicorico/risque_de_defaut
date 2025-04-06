import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres de simulation ---
N = 125                  # Nombre d'entreprises
T = 365                  # Horizon de temps en jours
dt = 30                  # Observation mensuelle
timesteps = np.arange(0, T + dt, dt)
n_steps = len(timesteps)
n_sim = 10000            # Nombre de simulations Monte Carlo

S0 = 100                 # Valeur initiale des entreprises
recovery_rate = 0.3      # Taux de recouvrement fixe
thresholds = [36, 50, 100]  # Seuils de défaut

# Volatilités selon les tranches
vols = np.zeros(N)
vols[:25] = 0.2
vols[25:50] = 0.25
vols[50:75] = 0.3
vols[75:100] = 0.35
vols[100:] = 0.5

# Reformater les vecteurs de volatilité pour le broadcasting
vols_broadcast = vols[np.newaxis, np.newaxis, :]  # shape: (1, 1, N)
timesteps_years = timesteps / 365  # Convertir en années
timesteps_broadcast = timesteps_years[np.newaxis, :, np.newaxis]  # shape: (1, n_steps, 1)

# Génération de trajectoires Browniennes indépendantes
np.random.seed(42)
Z = np.random.randn(n_sim, n_steps, N)
W = np.cumsum(Z * np.sqrt(dt / 365), axis=1)  # approximation de l'intégrale Brownienne

# Calcul des trajectoires S_i(t_k) pour chaque entreprise
S_paths = S0 * np.exp((-(vols_broadcast**2)/2) * timesteps_broadcast +
                      vols_broadcast * W)  # shape: (sim, time, company)

# Initialisation des résultats
results = {}

# Pour chaque seuil de défaut B
for B in thresholds:
    # On cherche la première date où S_i(t_k) < B
    default_mask = S_paths <= B
    default_time_idx = np.argmax(default_mask, axis=1)
    default_happened = np.any(default_mask, axis=1)

    # On enregistre l'indice temporel du défaut, ou -1 si pas de défaut
    default_time_idx[~default_happened] = -1

    # On récupère les prix S_i(τ_i) pour ceux qui ont fait défaut
    recovery_values = np.zeros(n_sim)
    for sim in range(n_sim):
        total_recovery = 0
        for i in range(N):
            idx = default_time_idx[sim, i]
            if idx != -1:
                total_recovery += recovery_rate * S_paths[sim, idx, i]
        recovery_values[sim] = total_recovery

    # Fonction de répartition de la dette Π*_T
    sorted_rec = np.sort(recovery_values)
    F_pi = np.arange(1, n_sim + 1) / n_sim

    results[B] = (sorted_rec, F_pi)

# Tracé des fonctions de répartition
plt.figure(figsize=(10, 6))
for B in thresholds:
    sorted_rec, F_pi = results[B]
    plt.plot(sorted_rec, F_pi, label=f"B = {B}")

plt.title("Fonction de répartition de la dette Π*_T pour différents seuils B (indépendants)")
plt.xlabel("Valeur de la dette Π*_T")
plt.ylabel("Fonction de répartition")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("a")

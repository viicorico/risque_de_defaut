import numpy as np
import matplotlib.pyplot as plt

# paramètres du modèle
nb_entreprises = 125
T = 1.0  # en années
N = 12  # 1 observation par mois
dt = T/N
val_init = 100
seuil_defaut = 50
volatilite = 0.4
taux_recouvrement = 0.3
Nmc = 1000

t = [k*dt for k in range(N+1)]

liste_defauts = []
liste_dettes = []

# simulations
for n in range(Nmc):
    defauts = 0
    dette = 0

    for _ in range(nb_entreprises):
        s = val_init
        a_defaut = False

        for i in t:
            z = np.random.randn()
            s = s*np.exp(-0.5*volatilite**2*dt + volatilite*np.sqrt(dt)*z)

            if not a_defaut and s <= seuil_defaut:
                defauts += 1
                dette += taux_recouvrement*s
                a_defaut = True

    liste_defauts.append(defauts)
    liste_dettes.append(dette)

defauts_arr = np.array(liste_defauts)
dettes_arr = np.array(liste_dettes)

# proba que nb de défauts >= k
valeurs_k = list(range(1,101))
probas = []

for k in valeurs_k:
    nb = np.sum(defauts_arr >= k)
    proba = nb/Nmc
    probas.append(proba)

# esperance de la dette sachant nb defauts > k
k_pour_plot = list(range(10,101,10))
esp_cond = []

for k in k_pour_plot:
    indices = defauts_arr > k
    if np.any(indices):
        esp = np.mean(dettes_arr[indices])
    else:
        esp = 0
    esp_cond.append(esp)

# fonction de repartition de la dette
x_det = np.linspace(0, max(dettes_arr), 200)
cdf = []

for x in x_det:
    cdf.append(np.mean(dettes_arr <= x))

# courbe 1
plt.figure()
plt.plot(valeurs_k, probas)
plt.xlabel("k")
plt.ylabel("P[nb défauts ≥ k]")
plt.title("probabilité d'au moins k défauts")
plt.grid()

# courbe 2
plt.figure()
plt.plot(k_pour_plot, esp_cond, marker='o')
plt.xlabel("k")
plt.ylabel("E[dette | nb défauts > k]")
plt.title("espérance de la dette conditionnelle")
plt.grid()

# courbe 3
plt.figure()
plt.plot(x_det, cdf)
plt.xlabel("x")
plt.ylabel("P[dette ≤ x]")
plt.title("fonction de répartition de la dette")
plt.grid()

plt.show()

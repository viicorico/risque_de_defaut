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

def simuler_S(Nmc):
    delta_t = T / N
    t = []
    for i in range(N + 1):
        t.append(i * delta_t)

    plt.figure(figsize=(10, 5))

    for _ in range(Nmc):
        W = generer_mouvement_brownien()
        S = []
        for i in range(N + 1):
            S_i = S0 * math.exp(-0.5 * sigma**2 * t[i] + sigma * W[i])
            S.append(S_i)
        plt.plot(t, S, alpha=0.5)

    plt.xlabel("Temps (t)")
    plt.ylabel("S(t)")
    plt.title(f"Simulation de {Nmc} trajectoires de S(t)")
    plt.show()

simuler_S(1000)

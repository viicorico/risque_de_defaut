# Projet MathFinance : Risque de Défaut d’Entreprise

## Description

Ce projet, réalisé dans le cadre du cursus Maths Finance 2024-2025 à CY TECH, porte sur la modélisation et la simulation du risque de défaut d’entreprise. L’objectif principal est d’évaluer, à l’aide de méthodes stochastiques et de simulations Monte Carlo, le risque de crédit (default risk) pour une seule entreprise ainsi que pour un groupe d’entreprises. Le projet inclut également l’estimation de la Value at Risk (VaR) et l’analyse de portefeuilles d’options.

## Contexte

Les crises financières et les désastres économiques (ex. : Metallgesellschaft, Lehman Brothers, etc.) démontrent l’importance de mesurer et anticiper le risque de défaut. La réglementation accrue (Solvabilité II, Bâle III) impose une modélisation rigoureuse des risques, notamment via des simulations stochastiques.

## Objectifs

- **Modélisation de la valeur des entreprises**  
  Utiliser un mouvement brownien géométrique pour simuler l’évolution de la valeur économique d’une entreprise.

- **Évaluation du risque de défaut**  
  Calculer, par Monte Carlo, la probabilité de faillite d’une entreprise et d’un groupe d’entreprises.

- **Calcul de la Value at Risk (VaR)**  
  Déterminer la VaR à différents horizons (ex. 10 jours, 365 jours) et pour divers niveaux de confiance (90 %, 99 %, voire 99.99 %) en utilisant :
  - La méthode de tri (ordonnement des pertes).
  - L’algorithme stochastique de Robbins-Monro.

- **Analyse de portefeuilles d’options**  
  Simuler l’évolution d’un portefeuille d’options et évaluer la VaR associée.

- **Étude des événements rares**  
  Appliquer des techniques d’échantillonnage préférentiel (Importance Sampling) et utiliser les théorèmes de Radon-Nicodym et de Girsanov pour estimer des probabilités d’événements extrêmes.

## Méthodologie et Techniques

- **Simulation Monte Carlo**  
  Génération de trajectoires pour modéliser la valeur d’une entreprise via un mouvement brownien géométrique.

- **Algorithme de Robbins-Monro**  
  Recherche du zéro d’une fonction d’espérance afin d’estimer la VaR.

- **Importance Sampling**  
  Technique permettant d’améliorer l’estimation des probabilités d’événements rares.

- **Théorèmes de Radon-Nicodym et de Girsanov**  
  Transformation des espaces de probabilité pour faciliter la simulation d’événements rares et l’évaluation du risque.

- **Modèle de risque de crédit**  
  Évaluation de la distribution des défauts et de la dette associée en cas de défaillance d’entreprises.

## Structure du Projet

- **Documentation**  
  - [`Slides_Projet_Risque_Defaut.pdf`](./Slides_Projet_Risque_Defaut.pdf) : Présente l’ensemble des concepts théoriques, des algorithmes et des références bibliographiques.

- **Code Source**  
  - Scripts de simulation (en Python) pour :
    - La simulation de la valeur d’entreprise.
    - Le calcul de la VaR par différentes méthodes (tri, Robbins-Monro).
    - L’implémentation des techniques d’importance sampling.
  - Modules dédiés à la simulation des portefeuilles d’options et à l’application de l’équation de Black-Scholes.

- **Résultats et Visualisations**  
  - Graphiques illustrant l’évolution des valeurs d’entreprise.
  - Fonctions de répartition des pertes.
  - Comparaison des méthodes de calcul de la VaR.

## Utilisation et Exécution

1. **Pré-requis**  
   - Installer les dépendances nécessaires (ex. : Python avec NumPy, matplotlib, etc. ou MATLAB selon l’implémentation).
   - Se référer au fichier de documentation pour comprendre les aspects théoriques du projet.

2. **Lancement des Simulations**  
   - Exécuter le script principal pour simuler la trajectoire d’une entreprise et calculer la VaR.
   - Pour la simulation d’un groupe d’entreprises, utiliser le module dédié en tenant compte des corrélations entre entreprises.

3. **Analyse des Résultats**  
   - Les résultats sont exportés sous forme de graphiques et d’analyses statistiques permettant de visualiser l’évolution des pertes et la distribution des défauts.
   - Les différentes méthodes de calcul (tri, Robbins-Monro, importance sampling) sont comparées afin de valider leur efficacité.

## Références

- **P. Glasserman**, *Monte Carlo Methods in Financial Engineering*, Springer Verlag.
- **Nicole El Karoui & Emmanuel Gobet**, *Les outils stochastiques des marchés financiers*.
- **R.G. Gallager**, MIT OpenCourseWare – *Discrete Stochastic Processes*.
- **S. Asmussen & B. Tuffin**, *Stochastic Simulation*.
- **J.A. Bucklew**, *Large Deviation Techniques in Decision, Simulation, and Estimation*.
- Autres références détaillées dans les slides.

## Contributeurs

- **Irina Kortchemski** – Encadrement théorique et référente pédagogique.
- **Équipe Projet** :
  - Emessinene Rachel
  -  Fernandes Valentin
  - Deines Edmond
  - Hang Victor
  - Dubois Jérémi



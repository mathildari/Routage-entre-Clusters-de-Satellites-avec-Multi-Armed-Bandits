# 🚀 Routage Satellite avec Multi-Armed Bandits (MAB)

## 📌 Description
Ce projet explore le **routage inter-cluster de nanosatellites** en utilisant des algorithmes de **Multi-Armed Bandits (MAB)**.  
Objectif : optimiser le chemin de transmission entre satellites en équilibrant **latence** et **consommation énergétique**.

## 🛰️ Contexte
- Réseau de nanosatellites organisés en **clusters**.  
- Chaque cluster possède **2 passerelles critiques** permettant la communication inter-cluster.  
- Le problème est modélisé comme un **bandit manchot** : chaque chemin est un bras, et l’algorithme apprend à choisir les meilleurs.

## ⚙️ Méthodes implémentées
- **Greedy** – exploitation pure (inefficace).  
- **ε-Greedy** – compromis exploration/exploitation.  
- **ε-decaying Greedy** – exploration décroissante, meilleure convergence.  
- **UCB** – borne de confiance supérieure, convergence rapide.  

## 📊 Résultats
- Greedy ≈ choix aléatoire.  
- ε-Greedy conserve une exploration trop forte.  
- ε-decaying Greedy & UCB atteignent les meilleures performances.  
- **UCB** converge le plus rapidement vers l’optimal.  

## 👩‍💻 Auteurs
- Daria Garnier  
- Loïc Capdeville  
- Alexy Fievet  
- Pierre Durollet  

Projet réalisé à l’**ENSEEIHT – Sciences du Numérique**.  

---

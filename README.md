# IA-Prediction-Impact-Environnemental

## Description

Cette IA a pour but de prédire l'impact sur le changement climatique (CO2 équivalent) d'un produit alimentaire. Le projet utilise des données de l'ADEME (Agribalyse 3.1) pour entraîner différents modèles de régression linéaire. Elle a été réalisée pour mettre en pratique mes apprentissages en Machine Learning.

## Données

Les données proviennent de l'ADEME et sont disponibles à [cette adresse](https://www.data.gouv.fr/datasets/agribalyse-3-1-synthese/). Le dataset contient 14 956 échantillons avec 32 variables incluant différents indicateurs d'impact environnemental.

### Variable cible

- **Changement climatique** : Impact en équivalent CO2

### Variables principales

- Score unique EF
- Particules fines
- Acidification terrestre et eaux douces
- Eutrophisation terrestre
- Utilisation du sol
- Épuisement des ressources eau
- Et autres indicateurs environnementaux

## Approche méthodologique

### 1. Analyse exploratoire

- Étude des corrélations entre variables
- Identification des variables les plus prédictives
- Nettoyage des données (gestion des valeurs manquantes)

### 2. Préprocessing

- **Pipeline numérique** : [`SimpleImputer`](main.ipynb) (stratégie médiane) + [`StandardScaler`](main.ipynb)
- **Pipeline catégoriel** : [`SimpleImputer`](main.ipynb) (valeur constante) + [`OneHotEncoder`](main.ipynb)
- Séparation train/test (80/20)

### 3. Modèles testés

Le notebook compare plusieurs algorithmes de régression :

- **LinearRegression** : Modèle de base sans régularisation
- **Ridge** : Régression Ridge (régularisation L2)
- **Lasso** : Régression Lasso (régularisation L1)
- **ElasticNet** : Combinaison Ridge + Lasso

### 4. Évaluation

Métriques utilisées :
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)  
- **R²** (Coefficient de détermination)

### API

L'API Flask dans le dossier [`api/`](api/) permet de :
- Faire des prédictions via des requêtes HTTP
- Déployer le modèle en production

# Réponses – Exploration du Midwest Survey

## 1. Combien d’exemples y a-t-il dans le dataset ?

Le dataset contient **2494 exemples**.  
Il comporte **28 variables (features)**.

---

## 2. Quelle est la distribution de la variable cible (target) ?

La variable cible est **Census_Region**.

Distribution (effectifs) :

- East North Central : 758 (30,4 %)
- West North Central : 358 (14,4 %)
- Middle Atlantic : 334 (13,4 %)
- South Atlantic : 248 (9,9 %)
- Pacific : 243 (9,7 %)
- Mountain : 190 (7,6 %)
- West South Central : 172 (6,9 %)
- East South Central : 97 (3,9 %)
- New England : 94 (3,8 %)

Le dataset est **déséquilibré**, car la région *East North Central* représente environ 30 % des observations, tandis que certaines régions représentent moins de 4 %.

---

## 3. Quelles sont les variables (features) utilisables pour prédire la cible ?

Le dataset contient 28 variables, notamment :

- RespondentID  
- What_would_you_call_the_part_of_the_country_you_live_in_now  
- How_much_do_you_personally_identify_as_a_Midwesterner  
- Plusieurs questions du type « Do you consider X state as part of the Midwest »  
- Gender  
- Age  
- Household_Income  
- Education  
- ZIP code  

La majorité des variables sont **catégorielles (texte/string)**.  
Seule la variable `RespondentID` est numérique.

---

## 4. Y a-t-il des valeurs manquantes dans le dataset ?

Il n’y a **aucune valeur manquante (NaN)** dans le dataset.

Cependant, certaines variables catégorielles peuvent contenir des réponses telles que :
- « Prefer not to answer »
- « Don't know »

Ces valeurs ne sont pas codées comme NaN mais peuvent représenter des données manquantes implicites.

---

## 5. Quelle est la réponse la plus fréquente à  
"How much do you personally identify as a Midwesterner" ?

Distribution :

- Not at all : 965  
- A lot : 697  
- Some : 528  
- Not much : 304  

La réponse la plus fréquente est :

**« Not at all »**

---

# Comparaison des modèles

## 6. Parmi les trois modèles, lequel a le meilleur recall ?

Recall pour la classe positive (« North Central ») :

- Logistic Regression : 0.0000  
- Random Forest : 0.9238  
- Gradient Boosting : 0.9895  

Le modèle ayant le meilleur recall est :

**Gradient Boosting**

Il identifie presque tous les vrais cas « North Central ».

---

## 7. Quel modèle a la meilleure application pratique ?

Structure de coûts :

- Faux positif (FP) = -10  
- Faux négatif (FN) = -1  
- Vrai positif (TP) = +5  
- Vrai négatif (TN) = +2  

Scores pratiques :

- Logistic Regression : 981  
- Random Forest : 4293  
- Gradient Boosting : 4629  

Le modèle le plus pertinent en pratique est :

**Gradient Boosting**

Il obtient le score pratique le plus élevé grâce à :
- Un recall très élevé
- Une très bonne précision
- Un excellent compromis global

---

## 8. Quel modèle généralise le mieux ?

Résultats de la validation croisée :

| Modèle | Accuracy train | Accuracy test | Écart |
|--------|---------------|--------------|--------|
| Logistic Regression | 0.5525 | 0.5525 | 0.0000 |
| Random Forest | 1.0000 | 0.9274 | 0.0726 |
| Gradient Boosting | 1.0000 | 0.9479 | 0.0521 |

- Plus petit écart train/test : **Logistic Regression**
- Plus grand écart (surapprentissage) : **Random Forest**

Techniquement, le modèle qui généralise le mieux (écart le plus faible) est :

**Logistic Regression**

Cependant, ses performances globales sont faibles.

---

## Choix final pour une application réelle

Si l’on devait choisir un modèle pour une application réelle :

**Gradient Boosting**

Raisons :
- Meilleur recall
- Meilleur score pratique
- Excellentes performances globales
- Écart train/test raisonnable

Marouane IZAM
Appogée: 23009631
Groupe:CAC2
<img src="Marouane_IZAM_CAC2_.JPG" style="height:264px;margin-right:232px"/>
# Projet : Analyse du jeu de données “Bank Marketing”

**Date de mise à disposition :** 13 février 2012  
**Source :** Campagnes de marketing direct d’une institution bancaire portugaise  

---

## 1. Contexte général du projet

Le projet **Bank Marketing** porte sur l’analyse et la modélisation de données issues de **campagnes de marketing téléphonique** menées par une **banque portugaise** entre **mai 2008 et novembre 2010**.  
L’objectif principal de ces campagnes était de **promouvoir un produit financier spécifique : le dépôt à terme (term deposit)**, c’est-à-dire un type de compte d’épargne à durée déterminée, offrant un taux d’intérêt fixe en échange de l’immobilisation des fonds pendant une période donnée.  

Les campagnes étaient menées **par téléphone**, et dans plusieurs cas, **plusieurs appels ont été nécessaires** pour contacter un même client ou confirmer son intérêt. Ces interactions ont permis de collecter un grand volume de données sur les **caractéristiques personnelles, professionnelles et financières** des clients, ainsi que sur les **conditions et résultats** des campagnes de marketing.  

Ce projet vise donc à **comprendre les facteurs influençant la décision du client** et à **développer un modèle de prédiction** permettant d’anticiper si un client acceptera (*yes*) ou refusera (*no*) de souscrire au dépôt à terme.  

---

## 2. Objectif du projet

L’objectif analytique principal consiste à construire un **modèle de classification supervisée** capable de prédire la variable cible **“y”**, qui indique si un client a souscrit (*yes*) ou non (*no*) au dépôt à terme.  
L’étude vise également à :  
- Identifier les **variables les plus déterminantes** dans la prise de décision du client,  
- Optimiser les **futurs efforts marketing** en ciblant les profils les plus réceptifs,  
- Évaluer les performances de différents **algorithmes de machine learning** (logistic regression, decision trees, random forest, SVM, etc.),  
- Réduire les **coûts et le nombre d’appels nécessaires** lors des campagnes futures.  

---

## 3. Caractéristiques du jeu de données

- **Nature des données :** Multivariées  
- **Domaine d’application :** Marketing bancaire et analyse comportementale  
- **Type de tâche :** Classification  
- **Types de variables :** Catégorielles (ex : emploi, état civil, type de contact) et numériques (ex : âge, durée d’appel, solde du compte)  
- **Nombre total d’observations :** 45 211  
- **Nombre de variables :** 16 principales (selon la version de la base utilisée)  

Les données sont représentatives d’une **population de clients bancaires portugais**, et regroupent à la fois des **informations socio-démographiques** (âge, emploi, niveau d’éducation), **financières** (solde du compte, crédits en cours, emprunts), et **contextuelles** (période de contact, durée de l’appel, conditions économiques).  

---

## 4. Détails sur les versions du dataset

Pour des raisons de performance et de reproductibilité, quatre versions du jeu de données sont mises à disposition :  

1. **bank-additional-full.csv**  
   - Contient **41 188 exemples** et **20 variables explicatives**.  
   - Les observations sont **classées chronologiquement** de **mai 2008 à novembre 2010**.  
   - Il s’agit de la version la plus complète et la plus fidèle à celle utilisée dans l’étude scientifique de *Moro et al., 2014*.  

2. **bank-additional.csv**  
   - Contient **4 119 exemples** (soit 10 % du précédent jeu), **sélectionnés aléatoirement**.  
   - Idéale pour des tests rapides ou des modèles nécessitant moins de calculs.  

3. **bank-full.csv**  
   - Ancienne version complète du jeu de données, avec **17 variables** (au lieu de 20).  
   - Les observations sont également triées par ordre chronologique.  

4. **bank.csv**  
   - Version échantillonnée aléatoirement (10 % de la précédente), avec **17 variables**.  
   - Conçue pour tester des **algorithmes plus lourds** comme les SVM ou les réseaux de neurones, nécessitant des temps de calcul réduits sur de petits volumes de données.  

---

## 5. Description de la variable cible (y)

- **Nom de la variable cible :** *y*  
- **Type :** Catégorique binaire  
- **Modalités possibles :**  
  - *yes* : le client a souscrit à un dépôt à terme  
  - *no* : le client n’a pas souscrit  

Cette variable représente le résultat final de la campagne, et constitue la **sortie** que le modèle doit prédire à partir des **caractéristiques des clients** et des **paramètres de la campagne**.  

---

## 6. Importance et applications

L’analyse de ce dataset revêt un **intérêt pratique majeur** pour le domaine bancaire et marketing. Elle permet de :  
- **Optimiser les stratégies de prospection** en ciblant uniquement les clients les plus susceptibles de répondre positivement,  
- **Améliorer le rendement des campagnes téléphoniques**, en réduisant le temps et le coût de contact,  
- **Fournir des recommandations stratégiques** basées sur les données, aidant à la prise de décision,  
- Et de manière plus générale, **illustrer l’impact du machine learning** dans le secteur financier, notamment dans la prédiction du comportement client.  
<img src="GRAPHE1_.PNG" style="height:264px;margin-right:232px"/>
<img src="GRAPHE2_.PNG" style="height:264px;margin-right:232px"/>
<img src="GRAPHE3_.PNG" style="height:264px;margin-right:232px"/>

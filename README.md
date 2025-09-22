# Data-Challenge-M2
Data challenge of the first semester of M2 math and IA at university paris saclay, a regression task and a classification task on tabular data.

code roadmap : 
we have the mains launchs files in bin.Project_launcher_xxx
we have the evalhooks in src.evaluation
we have the dataset loading and feature enginnering in src.dataset 
we have the training loop in src.training
we have the choice and setup of the models in src.models

each time we divide into two file with the spec for each task
in the base we create the base class with arguments and functions that are common to each task and in the classification and regression we create a class that heritate from the base class and has more functions and param according to the task



Brouillon :

DATA CHALLENGE : 

Methods list : 
Linear regression => RIDGE LASSO
regressio polynomiale (ou juste kernel linear reg)
Logistic regression 
GLM
GAM 
KNN
FOREST
LightGBM=> whatever that is
CatBoost => whatever that is
TREE
XGBOOST / BOOTSTRAPING
SVM
TIME SERIES autoregressif anova shit => yannig goude
KERNEL METHODS ON PRECEDENT ALGO
PERCEPTRONS MLP
FOUNDATION MODEL ON TABULAR DATA =PORTAL 

GRaph neural network ? Tabnet ? transformers for tabular


Feature engineering très important (géoloc, horaires, distance réelle, etc.)
Utilise des interactions non-linéaires
Encode les heures en sin/cos pour tenir compte de leur périodicité
Regroupe les zones géographiques avec du clustering (KMeans sur lat/lon)
Utilise cross-validation pour évaluer la robustesse
Attention aux outliers : les gros pourboires sont rares mais influents

Recommandation pour débuter
Baseline : Régression linéaire + quelques features simples
Ensembles : XGBoost / LightGBM avec tuning
Avancé : MLP avec features géographiques enrichies
Bonus : Analyse des pourboires extrêmes avec quantile regression


classif 
log reg multiclass softmx ou one vs rest
tree
forest
xgboost
svm
mlp
naive bayes 
knn
ensemble 
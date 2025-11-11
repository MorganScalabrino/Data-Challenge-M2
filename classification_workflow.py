from src.dataloader import HotelDataLoader
from src.models.classification import RandomForestModel, XGBoostModel, CatBoostModel, LogisticRegressionModel
from src.training.classification import train_and_evaluate, tune_hyperparameters
from src.evaluation.classification import generate_predictions, generate_predictions_ova
import os
from scipy.stats import randint, uniform
import numpy as np
from imblearn.over_sampling import SMOTE

model_name = "XGBoost"
SMOTTING=False


# Load data
dataloader = HotelDataLoader(os.path.join('..', 'classification_task', 'train_data.csv'), os.path.join('..', 'classification_task', 'test_data.csv'))
# X_train, X_val, y_train, y_val = dataloader.split_data()
X_train, y_train = dataloader.load_data()
X_train=dataloader.feature_engineering(X_train)
X_train = dataloader.preprocess_data(X_train)

# Initialize and train model
if model_name == "random_forest":
    model = RandomForestModel()
    param_grid={
                    'n_estimators': randint(50, 500),
                    'max_depth': [None, 10, 20, 30, 50, 70],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'max_features': ['auto', 'sqrt', 'log2', 0.5, 0.7],
                    # (optionally) bootstrap or not
                    'bootstrap': [True, False],
                    }
elif model_name == "XGBoost":
    model = XGBoostModel()
    param_grid =        param_space = {
                        'n_estimators': randint(50, 500),
                        'max_depth': randint(3, 15),
                        'learning_rate': uniform(0.01, 0.3),     
                        'subsample': uniform(0.5, 0.5),          
                        'colsample_bytree': uniform(0.5, 0.5),
                        'gamma': uniform(0, 5),
                        'reg_alpha': uniform(0, 5),
                        'reg_lambda': uniform(0.5, 5),
                        'min_child_weight': randint(1, 10),
                        }
elif model_name == "CatBoost":
    model = CatBoostModel()
    param_grid = {
                        'iterations': randint(100, 1000),
                        'depth': randint(4, 10),
                        'learning_rate': uniform(0.01, 0.3),
                        'l2_leaf_reg': uniform(1, 10),
                        'border_count': randint(32, 255),
                        'bagging_temperature': uniform(0, 1),
                        'random_strength': uniform(0, 10),
                        }
elif model_name == "LogisticRegression":
    model = LogisticRegressionModel()
    param_grid = {          'C': uniform(0.001, 10),     # inverse of regularization strength
                            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                            'l1_ratio': uniform(0, 1),   # only used when penalty='elasticnet'
                            'solver': ['saga', 'lbfgs']}
elif model_name == "OVA+Bin" :
    # one vs all + binary problem 
    # one vs all model no show vs all
    target_class = 2  
    X_train_ova = X_train.copy()
    y_train_ova = y_train.copy()
    y_train_ova = np.where(y_train_ova == target_class, 1, 0)
    if SMOTTING : # oversample the data
        X_train_ova, y_train_ova = SMOTE(random_state=42).fit_resample(X_train_ova, y_train_ova)

    # binary model cancellation vs check out
    mask = (y_train != target_class).astype(bool)
    mask_np=mask.to_numpy()
    X_train_bin = X_train[mask_np, :].copy() 
    y_train_bin = y_train[mask].copy()

    model_ova = RandomForestModel()
    #model_ova = RandomForestClassifier( n_estimators=300, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features='log2', bootstrap=True, class_weight={0:1,1:2}, random_state=42, n_jobs=-1)
    model_bin = RandomForestModel()
    param_grid={
                    'n_estimators': randint(50, 500),
                    'max_depth': [None, 10, 20, 30, 50, 70],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'max_features': ['auto', 'sqrt', 'log2', 0.5, 0.7],
                    # (optionally) bootstrap or not
                    'bootstrap': [True, False],
                    }
    best_model_ova = tune_hyperparameters(model_ova, param_grid, X_train_ova, y_train_ova, randomsearch=True, n_iter=40, n_jobs=-1)
    best_model_bin = tune_hyperparameters(model_bin, param_grid, X_train_bin, y_train_bin, randomsearch=True, n_iter=40, n_jobs=-1)


# Tune hyperparameters
best_model = tune_hyperparameters(model, param_grid, X_train, y_train, randomsearch=True, n_iter=40, n_jobs=-1)

# Generate predictions
if model_name == "OVA+Bin" :
    generate_predictions_ova(best_model_ova, best_model_bin, dataloader.test_path, dataloader.preprocessor, dataloader.feature_engineering, output_path=os.path.join('..', 'classification_task', "submission_",model_name,"_best.csv"))
else :
    generate_predictions(best_model, dataloader.test_path, dataloader.preprocessor, dataloader.feature_engineering, output_path=os.path.join('..', 'classification_task', "submission_",model_name,"_best.csv"))


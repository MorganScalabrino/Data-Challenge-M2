from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV, ParameterGrid
from tqdm import tqdm
import numpy as np

def train_and_evaluate(model, X_train, y_train, cv=5):
    cv_scores = []
    for train_idx, val_idx in tqdm(
        KFold(n_splits=cv).split(X_train, y_train),
        total=cv,
        desc="Cross-validating"
    ):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        model.fit(X_train_fold, y_train_fold)
        score = model.score(X_val_fold, y_val_fold)
        cv_scores.append(score)

    print(f"Cross-validated R² scores: {cv_scores}")
    print(f"Mean R²: {np.mean(cv_scores):.4f}")

    model.fit(X_train, y_train)
    return model

def tune_hyperparameters(model, param_grid, X_train, y_train, cv=5, randomsearch=True, n_iter=10):
    grid_search = RandomizedSearchCV(model, param_grid, cv=cv, scoring='r2', verbose=3, n_jobs=8, n_iter=n_iter) if randomsearch else GridSearchCV(model, param_grid, cv=cv, scoring='r2', verbose=3, n_jobs=-1) 
    with tqdm(total=len(list(ParameterGrid(param_grid))), desc="Grid search") as pbar:
        grid_search.fit(X_train, y_train)
        pbar.update(len(grid_search.cv_results_['params']))
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best R²: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

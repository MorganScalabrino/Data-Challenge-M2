from src.dataloader import SpotifyDataLoader
from src.models.regression import RandomForestModel, GradientBoostingModel
from src.training.regression import train_and_evaluate, tune_hyperparameters
from src.evaluation.regression import generate_predictions
import os

model = "gradient_boosting"

# Load data
dataloader = SpotifyDataLoader(os.path.join('..', 'regression_task', 'train_data.csv'), os.path.join('..', 'regression_task', 'test_data.csv'))
# X_train, X_val, y_train, y_val = dataloader.split_data()
X_train, y_train = dataloader.load_data()
X_train = dataloader.preprocess_data(X_train)

# Initialize and train model
if model == "random_forest":
    model = RandomForestModel()
    #param_grid = {'n_estimators': [500], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10, 15]}
    param_grid = {'n_estimators': [500], 'max_depth': [None, 20], 'min_samples_split': [10]}
elif model == "gradient_boosting":
    model = GradientBoostingModel()
    param_grid = {'n_estimators': [300, 350, 400], 'learning_rate' : [0.05, 0.001], 'max_depth': [None, 20], 'min_samples_split': [5, 7, 10], 'subsample': [0.8, 0.7, 0.6]}

# Tune hyperparameters
best_model = tune_hyperparameters(model, param_grid, X_train, y_train, randomsearch=True, n_iter=20)

# Generate predictions
generate_predictions(best_model, dataloader.test_path, dataloader.preprocessor, output_path=os.path.join('..', 'regression_task', 'submission_gb_1.csv'))

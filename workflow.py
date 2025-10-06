from src.dataloader import DataLoader
from src.models.regression import RandomForestModel
from src.training.regression import train_and_evaluate, tune_hyperparameters
from src.evaluation.regression import generate_predictions
import os

# Load data
dataloader = DataLoader(os.path.join('..', 'regression_task', 'train_data.csv'), os.path.join('..', 'regression_task', 'test_data.csv'))
X_train, X_val, y_train, y_val = dataloader.split_data()

# Initialize and train model
model = RandomForestModel()
#model = train_and_evaluate(model, X_train, y_train)

# Tune hyperparameters
#param_grid = {'n_estimators': [500], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10, 15]}
param_grid = {'n_estimators': [500], 'max_depth': [None, 20], 'min_samples_split': [10]}
best_model = tune_hyperparameters(model, param_grid, X_train, y_train)

# Generate predictions
generate_predictions(best_model, dataloader.test_path, dataloader.preprocessor, output_path=os.path.join('..', 'regression_task', 'submission_1.csv'))

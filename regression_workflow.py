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
    param_grid = {'n_estimators': [500], 'max_depth': [None, 10, 20], 'max_features': [3, 4], 'min_samples_split': [2, 5, 10, 15]}
elif model == "gradient_boosting":
    model = GradientBoostingModel()
    # param_grid = {'n_estimators': [300, 350, 400, 500], 'learning_rate' : [0.05, 0.07, 0.1], 'max_depth': [30, 20, 15, 10, 5, 3], 'min_samples_split': [2], 'subsample': [0.8, 0.7, 0.6]}
    param_grid = {'n_estimators': [400, 450, 500], 'learning_rate' : [0.05], 'max_depth': [30, 20], 'min_samples_split': [2], 'subsample': [0.9, 0.8]}

# Tune hyperparameters
best_model = tune_hyperparameters(model, param_grid, X_train, y_train, randomsearch=True, n_iter=40, n_jobs=-1)

# Generate predictions
generate_predictions(best_model, dataloader.test_path, dataloader.preprocessor, output_path=os.path.join('..', 'regression_task', 'submission_gb_6_noonehot_best.csv'))


# gb_2
# Best parameters: {'subsample': 0.6, 'n_estimators': 350, 'min_samples_split': 10, 'max_depth': None, 'learning_rate': 0.05}
# Best R²: 0.4737

# gb_3
# Best parameters: {'subsample': 0.7, 'n_estimators': 350, 'min_samples_split': 2, 'max_depth': 10, 'learning_rate': 0.1}
# Best R²: 0.4293

# gb_4 no one hot encoding
# Best parameters: {'subsample': 0.7, 'n_estimators': 400, 'min_samples_split': 2, 'max_depth': 10, 'learning_rate': 0.1}
# Best R²: 0.4585

# gb_5 no one hot encoding
# Best parameters: {'subsample': 0.8, 'n_estimators': 400, 'min_samples_split': 2, 'max_depth': 15, 'learning_rate': 0.05}
# Best R²: 0.5010

# gb_6 no one hot encoding
# Best parameters: {'subsample': 0.8, 'n_estimators': 400, 'min_samples_split': 2, 'max_depth': 20, 'learning_rate': 0.05}
# Best R²: 0.4977
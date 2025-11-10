from src.dataloader import SpotifyDataLoader
from src.models.regression import RandomForestModel, GradientBoostingModel
from src.training.regression import train_and_evaluate, tune_hyperparameters
from src.evaluation.regression import generate_predictions
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import scipy.stats as stats

# Load data
dataloader = SpotifyDataLoader(os.path.join('..', 'regression_task', 'train_data.csv'), os.path.join('..', 'regression_task', 'test_data.csv'))
# X_train, X_val, y_train, y_val = dataloader.split_data()
X_train, X_val, y_train, y_val = dataloader.split_data()

# Initialize and train model
model = GradientBoostingModel(n_estimators = 400, learning_rate = 0.05, max_depth = 15, min_samples_split = 2, subsample = 0.8)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)


r2 = r2_score(y_val, y_pred)
print(f"R² Score: {r2:.4f}")

# Comparison of densities for val and pred
plt.figure()
sns.histplot(y_val, color='blue', label='y_val (True)', kde=True, stat='density', alpha=0.5)
sns.histplot(y_pred, color='red', label='y_pred (Predicted)', kde=True, stat='density', alpha=0.5)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Distribution Comparison: validation vs prediction')
plt.legend()
plt.show()

# Q-Q plot for val and pred popularity distribution
plt.figure()
stats.probplot(y_val, dist="norm", plot=plt)
stats.probplot(y_pred, dist="norm", plot=plt)
plt.title('Q-Q Plot: y_val vs y_pred')
plt.legend(['y_val (True)', 'y_pred (Predicted)'])
plt.show()

# KDE
kde_val = stats.gaussian_kde(y_val)
kde_pred = stats.gaussian_kde(y_pred)

x_grid = np.linspace(0, 100, 1000)
pdf_val = kde_val(x_grid)
pdf_pred = kde_pred(x_grid)

# Compute KL Divergence
kl_divergence = np.sum(np.where(pdf_val != 0, pdf_val * np.log(pdf_val / pdf_pred), 0))
print(f"KL Divergence: {kl_divergence:.4f}")

# R² Score: 0.4990
# KL Divergence: 1.3077
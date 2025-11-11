# Data-Challenge-M2
Data challenge of the first semester of M2 math and IA at University Paris-Saclay, consisting in two prediction tasks on tabular data : a regression task to predict spotify tracks popularity from various features (genre, danceability, mode...) and a classification task to predict wether a reservation in a hotel will be canceled or not.

## Datasets
Train and test datasets for classification and regression tasks are supposed to be in two folders at the same level as this repository, named classification_task and regression_task, respectively.

## Code roadmap

### Main functions
- Training, tuning and results obtention workflows for the two tasks can be found in classification_workflow.py and regression_workflow.py
- Function to produce predictions on test set can be found in src.evaluation
- DataLoaders with feature preprocessing and visualization methods are in dataloader.py (for both tasks)
- Training and fine tuning functions are defined in src.training
- Used models are defined in src.models

We divide each one of these folders in two file (classification.py and regression.py) with the specific capacities for each task.

### Data exploration and results analysis
- In data_exploration_classification.ipynb you can find the vizualisation of the classification dataset AND the vizualisation of the one vs all + binary model at the end of the file, you can also find just before that the greedy search algorithm to visualize the importance of the features via their positive or negative impact on the score.
- In data_visualization_regression.ipynb, you will find visualization of the spotify dataset, correlations, popularity density plots, dimension reduction...
- In regression_test_residuals.py you will be able to plot distributions of predicted popularity against real popularity on a validation test, for the best model we have. These are compared with a Q-Q plot and we compute KL divergence.

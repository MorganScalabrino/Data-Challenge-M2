# Data-Challenge-M2
Data challenge of the first semester of M2 math and IA at university paris saclay, a regression task and a classification task on tabular data.

code roadmap : 
we have the mains launchs files in classification/regression _workflow
we have the evalhooks in src.evaluation
we have the dataset loading and feature enginnering in src.dataset 
we have the training and fine tuning in src.training
we have the choice and setup of the models in src.models

each time we divide into two file (classification and regression) with the specific capacities for each task

In Data_exploration_classification.py you can find the vizualisation of the classification dataset AND the vizualisation of the one vs all + binary model at the end of the file, you can also find just before that the greedy search algorithm to visualize the importance of the features via their positive or negative impact on the score.

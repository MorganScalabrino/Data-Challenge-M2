from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

from sklearn.ensemble import VotingRegressor, StackingRegressor

class BaseModelWrapper(BaseEstimator, RegressorMixin):
    """class to wrap models to allow the use of sklearn utilities
    """
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self
    
class RandomForestModel(BaseModelWrapper):
    def __init__(self, n_estimators=300, random_state=42, **kwargs):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, **kwargs)
        super().__init__(model)

class XGBoostModel(BaseModelWrapper):
    def __init__(self, n_estimators=100, random_state=42, **kwargs):
        model = xgb(n_estimators=n_estimators, random_state=random_state, **kwargs)
        super().__init__(model)

class LogisticRegressionModel(BaseModelWrapper):
    def __init__(self, **kwargs):
        model = LogisticRegression(**kwargs)
        super().__init__(model)

class CatBoostModel(BaseModelWrapper):
    def __init__(self, n_estimators=100, random_state=42, **kwargs):
        model = CatBoostClassifier(n_estimators=n_estimators, random_state=random_state, **kwargs)
        super().__init__(model)

# Ensemble Methods
class VotingEnsemble(BaseModelWrapper):
    def __init__(self, estimators, **kwargs):
        model = VotingRegressor(estimators=estimators, **kwargs)
        super().__init__(model)


class StackingEnsemble(BaseModelWrapper):
    def __init__(self, estimators, final_estimator, **kwargs):
        model = StackingRegressor(estimators=estimators, final_estimator=final_estimator, **kwargs)
        super().__init__(model)
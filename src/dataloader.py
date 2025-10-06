import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataLoader:
    def __init__(self, train_path, test_path=None):
        self.train_path = train_path
        self.test_path = test_path
        self.preprocessor = self._build_preprocessor()

    def _build_preprocessor(self):
        # Define numerical and categorical features
        numerical_features = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        categorical_features = ['track_genre']

        # Create transformers
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        return preprocessor

    def load_data(self):
        # Load and preprocess data
        train = pd.read_csv(self.train_path)
        X = train.drop(columns=['popularity', 'Unnamed: 0', 'row_id'])
        y = train['popularity']
        X = self.preprocessor.fit_transform(X)
        return X, y

    def split_data(self, test_size=0.2, random_state=42):
        X, y = self.load_data()
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

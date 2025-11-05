import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.decomposition import PCA, SparsePCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.pipeline import Pipeline, FunctionTransformer

class SpotifyDataLoader:
    def __init__(self, train_path, test_path=None):
        self.train_path = train_path
        self.test_path = test_path
        self.preprocessor = self._build_preprocessor()

    def _build_preprocessor(self):
        # Define numerical and categorical features
        numerical_features = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        categorical_features = ['key', 'mode', 'time_signature', 'track_genre']
        boolean_features = ['explicit']

        # Create transformers
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        boolean_transformer = StandardScaler()

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features),
                ('bool', boolean_transformer, boolean_features)
            ]
        )
        return preprocessor

    def load_data(self):
        # Load and preprocess data
        train = pd.read_csv(self.train_path)
        X = train.drop(columns=['popularity', 'Unnamed: 0', 'row_id'])
        y = train['popularity']
        return X, y
    
    def preprocess_data(self, X):
        return self.preprocessor.fit_transform(X)
    
    def apply_pca(self, X, n_components=2):
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('pca', PCA(n_components=n_components))
        ])
        return pipeline.fit_transform(X), pipeline
    
    def apply_sparse_pca(self, X, n_components=2):
        to_dense = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('to_dense', to_dense),
            ('sparse_pca', SparsePCA(n_components=n_components))
        ])
        return pipeline.fit_transform(X), pipeline
    
    def apply_tsne(self, X, n_components=2):
        to_dense = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('to_dense', to_dense),
            ('tsne', TSNE(n_components=n_components))
        ])
        return pipeline.fit_transform(X), pipeline
    
    def apply_umap(self, X, n_components=2):
        X_preprocessed = self.preprocessor.fit_transform(X)
        umap = UMAP(n_components=n_components)
        return umap.fit_transform(X_preprocessed), umap

    def split_data(self, test_size=0.2, random_state=42):
        X, y = self.load_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train = self.preprocess_data(X_train)
        X_val = self.preprocessor.transform(X_val)
        return X_train, X_val, y_train, y_val

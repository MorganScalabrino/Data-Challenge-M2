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
        self.numerical_features = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'key', 'tempo', 'mode']
        self.categorical_features = ['time_signature', 'track_genre']
        self.boolean_features = ['explicit']

        # Create transformers
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        boolean_transformer = StandardScaler()

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                # ('cat', categorical_transformer, self.categorical_features),
                ('bool', boolean_transformer, self.boolean_features)
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
    
    def apply_pca(self, X, n_components=2, by=None):
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('pca', PCA(n_components=n_components))
        ])
        red_dict = {}
        if by:
            categories = X[by].unique()
            print(categories)
            red_dict = {}
            for category in categories:
                X_filtered = X[X[by] == category]
                red_dict[category] = {"X" : X_filtered, "X_red" : pipeline.fit_transform(X_filtered), "filter" : X[by]==category, "pipeline" : pipeline}
        else:
            red_dict["not filtered"] = {"X" : X, "X_red" : pipeline.fit_transform(X), "filter" : True, "pipeline" : pipeline}
        return pipeline.fit_transform(X), pipeline
    
    def apply_sparse_pca(self, X, n_components=2):
        to_dense = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            #('to_dense', to_dense),
            ('sparse_pca', SparsePCA(n_components=n_components))
        ])
        return pipeline.fit_transform(X), pipeline
    
    def apply_tsne(self, X, n_components=2, by=None, except_col=[]):
        # Create transformers
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        boolean_transformer = StandardScaler()

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, list(set(self.numerical_features)-set(except_col))),
                # ('cat', categorical_transformer, list(set(self.categorical_features)-set(except_col))),
                ('bool', boolean_transformer, list(set(self.boolean_features)-set(except_col)))
            ]
        )
        # to_dense = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            # ('to_dense', to_dense),
            ('tsne', TSNE(n_components=n_components))
        ])
        red_dict = {}
        if by:
            categories = X[by].unique()
            print(categories)
            red_dict = {}
            for category in categories[:3]:
                X_filtered = X[X[by] == category]
                X_filtered.drop(columns=[by])
                red_dict[category] = {"X" : X_filtered, "X_red" : pipeline.fit_transform(X_filtered), "filter" : X[by]==category, "pipeline" : pipeline}
        else:
            red_dict["not filtered"] = {"X" : X, "X_red" : pipeline.fit_transform(X), "filter" : [True]*len(X), "pipeline" : pipeline}
        return red_dict
    
    def apply_umap(self, X, n_components=2, by=None, except_col=[]):
        # Create transformers
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        boolean_transformer = StandardScaler()

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, list(set(self.numerical_features)-set(except_col))),
                # ('cat', categorical_transformer, list(set(self.categorical_features)-set(except_col))),
                ('bool', boolean_transformer, list(set(self.boolean_features)-set(except_col)))
            ]
        )

        red_dict = {}
        if by:
            categories = X[by].unique()
            print(categories)
            red_dict = {}
            for category in categories[:3]:
                X_filtered = X[X[by] == category]
                X_filtered.drop(columns=[by])
                X_filtered = preprocessor.fit_transform(X_filtered)
                umap = UMAP(n_components=n_components)
                red_dict[category] = {"X" : X_filtered, "X_red" : umap.fit_transform(X_filtered), "filter" : X[by]==category, "pipeline" : umap}
        else:
            X_preprocessed = preprocessor.fit_transform(X)
            umap = UMAP(n_components=n_components)
            red_dict["not filtered"] = {"X" : X, "X_red" : umap.fit_transform(X_preprocessed), "filter" : [True]*len(X), "pipeline" : umap}
        return red_dict

    def split_data(self, test_size=0.2, random_state=42):
        X, y = self.load_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train = self.preprocess_data(X_train)
        X_val = self.preprocessor.transform(X_val)
        return X_train, X_val, y_train, y_val




class HotelDataLoader:
    def __init__(self, train_path, test_path=None):
        self.train_path = train_path
        self.test_path = test_path
        self.preprocessor = self._build_preprocessor()

    def _build_preprocessor(self):
        # Define numerical and categorical features
        self.numerical_features = ['lead_time', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list', 'adr', 'required_car_parking_spaces', 'total_of_special_requests']
        self.categorical_features =['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type']

        # Create transformers
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features),
            ]
        )
        return preprocessor

    def load_data(self):
        # Load and preprocess data
        train = pd.read_csv(self.train_path)
        X = train.drop(columns=['reservation_status', 'Unnamed: 0', 'row_id'])
        y = train['reservation_statu']
        return X, y
    
    def preprocess_data(self, X):
        return self.preprocessor.fit_transform(X)
    
    def apply_pca(self, X, n_components=2, by=None):
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('pca', PCA(n_components=n_components))
        ])
        red_dict = {}
        if by:
            categories = X[by].unique()
            print(categories)
            red_dict = {}
            for category in categories:
                X_filtered = X[X[by] == category]
                red_dict[category] = {"X" : X_filtered, "X_red" : pipeline.fit_transform(X_filtered), "filter" : X[by]==category, "pipeline" : pipeline}
        else:
            red_dict["not filtered"] = {"X" : X, "X_red" : pipeline.fit_transform(X), "filter" : True, "pipeline" : pipeline}
        return pipeline.fit_transform(X), pipeline
    
    def apply_sparse_pca(self, X, n_components=2):
        to_dense = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            #('to_dense', to_dense),
            ('sparse_pca', SparsePCA(n_components=n_components))
        ])
        return pipeline.fit_transform(X), pipeline
    
    def apply_tsne(self, X, n_components=2, by=None, except_col=[]):
        # Create transformers
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, list(set(self.numerical_features)-set(except_col))),
                ('cat', categorical_transformer, list(set(self.categorical_features)-set(except_col))),
            ]
        )
        # to_dense = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            # ('to_dense', to_dense),
            ('tsne', TSNE(n_components=n_components))
        ])
        red_dict = {}
        if by:
            categories = X[by].unique()
            print(categories)
            red_dict = {}
            for category in categories[:3]:
                X_filtered = X[X[by] == category]
                X_filtered.drop(columns=[by])
                red_dict[category] = {"X" : X_filtered, "X_red" : pipeline.fit_transform(X_filtered), "filter" : X[by]==category, "pipeline" : pipeline}
        else:
            red_dict["not filtered"] = {"X" : X, "X_red" : pipeline.fit_transform(X), "filter" : [True]*len(X), "pipeline" : pipeline}
        return red_dict
    
    def apply_umap(self, X, n_components=2, by=None, except_col=[]):
        # Create transformers
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, list(set(self.numerical_features)-set(except_col))),
                ('cat', categorical_transformer, list(set(self.categorical_features)-set(except_col))),
            ]
        )

        red_dict = {}
        if by:
            categories = X[by].unique()
            print(categories)
            red_dict = {}
            for category in categories[:3]:
                X_filtered = X[X[by] == category]
                X_filtered.drop(columns=[by])
                X_filtered = preprocessor.fit_transform(X_filtered)
                umap = UMAP(n_components=n_components)
                red_dict[category] = {"X" : X_filtered, "X_red" : umap.fit_transform(X_filtered), "filter" : X[by]==category, "pipeline" : umap}
        else:
            X_preprocessed = preprocessor.fit_transform(X)
            umap = UMAP(n_components=n_components)
            red_dict["not filtered"] = {"X" : X, "X_red" : umap.fit_transform(X_preprocessed), "filter" : [True]*len(X), "pipeline" : umap}
        return red_dict

    def split_data(self, test_size=0.2, random_state=42):
        X, y = self.load_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train = self.preprocess_data(X_train)
        X_val = self.preprocessor.transform(X_val)
        return X_train, X_val, y_train, y_val


    def feature_engineering (self, X): 
        # categorisation of the year variable
        X['arrival_date_year'] = X['arrival_date_year'].astype('category')

        # new variables
        off_season=["February", "December", "November", "January"] 
        X["off_season"]=  X["arrival_date_month"].isin(off_season).astype(int)
        X["expensive"]=(X['adr'] > X['adr'].mean()+X['adr'].std()).astype(int)
        X["cheap"]=(X['adr'] > X['adr'].mean()+X['adr'].std()).astype(int)
        X["weekend"]=((X['stays_in_week_nights'] == 0) & (X['stays_in_weekend_nights']>0)).astype(int)

        # get 0 2 1 =90+%  3 and higher as one  and get 0 1 2 3 4 5 =90+%  6 and higher as one
        X["long_wk_stays"]=(X['stays_in_week_nights']>=6).astype(int) 
        X["long_wkd_stays"]=(X['stays_in_weekend_nights']>=3).astype(int) 

        # COS SIN Transformation for cyclical Variables 
        month_map = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }
        X["arrival_date_month_x"] = np.sin(2 * np.pi * X["arrival_date_month"].map(month_map) / 12)
        X["arrival_date_month_y"] = np.cos(2 * np.pi * X["arrival_date_month"].map(month_map) / 12)
        X["arrival_date_week_number_x"] = np.sin(2 * np.pi * X["arrival_date_week_number"] / 52)
        X["arrival_date_week_number_y"] = np.cos(2 * np.pi * X["arrival_date_week_number"] / 52)
        X["arrival_date_day_of_month_x"] = np.sin(2 * np.pi * X["arrival_date_day_of_month"] / 31)
        X["arrival_date_day_of_month_y"] = np.cos(2 * np.pi * X["arrival_date_day_of_month"] / 31)

        X.drop('arrival_date_month', axis=1, inplace=True)
        X.drop('arrival_date_week_number', axis=1, inplace=True)
        X.drop('arrival_date_day_of_month', axis=1, inplace=True)

        return X




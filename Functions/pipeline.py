
import numpy as np
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder
from fixed_params import random_state, decimal_places, imputer_max_iter, categorical_features

# todo: options for meta and without

def construct_pipelines(numeric_features_index, categorical_features_index):
    # define stages of pipeline
    scaler = StandardScaler()
    random_forest = RandomForestClassifier()
    oh_encoder = OneHotEncoder(handle_unknown='error', drop="if_binary")
    gradient_boosting = GradientBoostingClassifier(validation_fraction=0.1, warm_start=True)

    # Bayesian ridge is quicker/simpler than RF:
    imp_iter_num = IterativeImputer(missing_values=np.nan, max_iter=imputer_max_iter,
                                    random_state=random_state)

    imp_iter_cat = IterativeImputer(estimator=RandomForestClassifier(),
                                    initial_strategy='most_frequent',
                                    missing_values=np.nan,
                                    max_iter=1,
                                    random_state=random_state,
                                    )
    # Preprocessor pipelines:
    categorical_transformer = Pipeline(
        steps=[("imputer", imp_iter_cat), ("oh_encoder", oh_encoder)]
    )

    numeric_transformer = Pipeline(
        steps=[("imputer", imp_iter_num), ("scaler", scaler)]
    )

    preprocessor = ColumnTransformer(sparse_threshold=0,
        transformers=[
            ("num", numeric_transformer, numeric_features_index),
            ("cat", categorical_transformer, categorical_features_index),
        ]
    )

    # GB preprocessor pipeline:
    categorical_transformer_GB = Pipeline(
        steps=[("oh_encoder", oh_encoder)]
    )

    numeric_transformer_GB = Pipeline(
        steps=[("scaler", scaler)]
    )

    preprocessor_GB = ColumnTransformer(sparse_threshold=0,
        transformers=[
            ("num", numeric_transformer_GB, numeric_features_index),
            ("cat", categorical_transformer_GB, categorical_features_index),
        ]
    )

    # full pipelines:

    pipe_rf = Pipeline([
        ("preprocessor", preprocessor), ('classifier', random_forest)
    ])

    pipe_gb = Pipeline([
        ("preprocessor", preprocessor_GB), ('classifier', gradient_boosting)
    ])
    return pipe_rf, pipe_gb
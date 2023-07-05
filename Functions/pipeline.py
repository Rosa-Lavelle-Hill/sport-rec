
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from fixed_params import random_state, decimal_places, imputer_max_iter, categorical_features, outcome
from imblearn.pipeline import Pipeline as imbpipeline


def construct_pipelines(numeric_features_index, categorical_features_index, multi_label):
    # define stages of pipeline
    scaler = StandardScaler()
    random_forest = RandomForestClassifier()
    oh_encoder = OneHotEncoder(handle_unknown='error', drop="if_binary")
    gradient_boosting = GradientBoostingClassifier(validation_fraction=0.1, warm_start=True)
    enet = LogisticRegression(penalty="elasticnet", random_state=random_state, solver="saga")
    log = LogisticRegression(random_state=random_state, solver="newton-cg")

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

    if multi_label == True:

        pipe_log = imbpipeline(steps=[['preprocessor', preprocessor],
                                      ['ml', MultiOutputClassifier(estimator=log)]])

        pipe_enet = imbpipeline(steps=[['preprocessor', preprocessor],
                                       ['ml', MultiOutputClassifier(estimator=enet)]])

        pipe_rf = imbpipeline(steps=[['preprocessor', preprocessor],
                                     ['ml', MultiOutputClassifier(estimator=random_forest)]])

        pipe_gb = imbpipeline(steps=[['preprocessor', preprocessor],
                                     ['ml', MultiOutputClassifier(estimator=gradient_boosting)]])
    else:
        pipe_log = Pipeline([
            ("preprocessor", preprocessor),
            ('classifier', log)
        ])

        pipe_enet = Pipeline([
            ("preprocessor", preprocessor),
            ('classifier', enet)
        ])

        pipe_rf = Pipeline([
            ("preprocessor", preprocessor),
            ('classifier', random_forest)
        ])

        pipe_gb = Pipeline([
            ("preprocessor", preprocessor_GB),
            ('classifier', gradient_boosting)
        ])
    return pipe_log, pipe_enet, pipe_rf, pipe_gb



def construct_smote_pipelines(numeric_features_index, categorical_features_index, multi_label):
    # define stages of pipeline
    scaler = StandardScaler()
    oh_encoder = OneHotEncoder(handle_unknown='error', drop="if_binary")
    random_forest = RandomForestClassifier()
    gradient_boosting = GradientBoostingClassifier(validation_fraction=0.1, warm_start=True)
    enet = LogisticRegression(penalty="elasticnet", random_state=random_state, solver="saga")
    log = LogisticRegression(random_state=random_state, solver="newton-cg", penalty=None)

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
    smote = SMOTE(random_state=random_state)
    if multi_label == True:

        pipe_log = imbpipeline(steps=[['preprocessor', preprocessor],
                                      ['smote', smote],
                                      ['ml', MultiOutputClassifier(estimator=log)]])

        pipe_enet = imbpipeline(steps=[['preprocessor', preprocessor],
                                       ['smote', smote],
                                       ['ml', MultiOutputClassifier(estimator=enet)]])

        pipe_rf = imbpipeline(steps=[['preprocessor', preprocessor],
                                     ['smote', smote],
                                     ['ml', MultiOutputClassifier(estimator=random_forest)]])

        pipe_gb = imbpipeline(steps=[['preprocessor', preprocessor],
                                     ['smote', smote],
                                     ['ml', MultiOutputClassifier(estimator=gradient_boosting)]])

    else:

        pipe_log = imbpipeline(steps=[['preprocessor', preprocessor],
                                ['smote', smote],
                                ['classifier', log]])

        pipe_enet = imbpipeline(steps=[['preprocessor', preprocessor],
                                ['smote', smote],
                                ['classifier', enet]])

        pipe_rf = imbpipeline(steps=[['preprocessor', preprocessor],
                                ['smote', smote],
                                ['classifier', random_forest]])

        pipe_gb = imbpipeline(steps=[['preprocessor', preprocessor],
                                ['smote', smote],
                                ['classifier', gradient_boosting]])
    return pipe_log, pipe_enet, pipe_rf, pipe_gb



def construct_dummy_pipelines(numeric_features_index, categorical_features_index):
    # define stages of pipeline
    scaler = StandardScaler()
    oh_encoder = OneHotEncoder(handle_unknown='error', drop="if_binary")
    dummy_mf = DummyClassifier(strategy="most_frequent")
    dummy_rand = DummyClassifier(strategy="uniform")
    dummy_strat = DummyClassifier(strategy="stratified")


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

    pipe_dum_mf = Pipeline([
        ("preprocessor", preprocessor),
        ('classifier', dummy_mf)
    ])

    pipe_dum_random = Pipeline([
        ("preprocessor", preprocessor),
        ('classifier', dummy_rand)
    ])

    pipe_dum_strat = Pipeline([
        ("preprocessor", preprocessor),
        ('classifier', dummy_strat)
    ])

    return pipe_dum_mf, pipe_dum_random, pipe_dum_strat
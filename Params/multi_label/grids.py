nfolds = 5
rf_params = {"ml__estimator__min_samples_split": [2, 4, 6],
             "ml__estimator__max_depth": [5, 10, 15],
             "ml__estimator__n_estimators": [500],
             "ml__estimator__random_state": [93],
             "ml__estimator__max_features": [0.8, 1]}

gb_params = {"ml__estimator__min_samples_leaf": [2, 4, 6],
             "ml__estimator__max_depth": [5, 10, 15],
             "ml__estimator__random_state": [93],
             "ml__estimator__max_iter": [500],
             "ml__estimator__learning_rate": [0.01, 0.1, 1]}

log_params = {"ml__estimator__penalty": [None]}

enet_params = {"ml__estimator__C":[0.001, 0.01, 0.5, 1], # Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization.
              "ml__estimator__l1_ratio": [0.2, 0.5, 0.8],
              "ml__estimator__max_iter": [1000]}

# smote ml grids
rf_params_sm = {"estimator__classifier__min_samples_split": [2, 4, 6],
             "estimator__classifier__max_depth": [5, 10, 15],
             "estimator__classifier__n_estimators": [500],
             "estimator__classifier__random_state": [93],
             "estimator__classifier__max_features": [0.8, 1],
             "estimator__oversampler__k_neighbors": [3, 5, 7],
             "estimator__oversampler__sampling_strategy": [0.5, 1.0, 'auto']
}

gb_params_sm = {"estimator__classifier__min_samples_leaf": [2, 4, 6],
                "estimator__classifier__max_depth": [5, 10],
                "estimator__classifier__random_state": [93],
                "estimator__classifier__n_estimators": [300, 500, 800],
                "estimator__classifier__max_features": [0.7, None],
                "estimator__classifier__learning_rate": [0.01, 0.05, 0.1],
                "estimator__oversampler__k_neighbors": [3, 5],
                "estimator__oversampler__sampling_strategy": [0.5, 1.0],
                "estimator__classifier__subsample": [0.8, 1.0],
                }

log_params_sm = {"estimator__classifier__penalty": [None]}

enet_params_sm = {"estimator__classifier__C":[0.001, 0.01, 0.5, 1], # Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization.
              "estimator__classifier__l1_ratio": [0.2, 0.5, 0.8],
              "estimator__classifier__max_iter": [1000]}
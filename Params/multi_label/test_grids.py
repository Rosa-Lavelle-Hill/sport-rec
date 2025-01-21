rf_params = {'ml__estimator__max_depth': [5],
             'ml__estimator__max_features': [0.3],
             'ml__estimator__min_samples_split': [3],
             'ml__estimator__n_estimators': [5],
             'ml__estimator__random_state': [93]}

gb_params = {"ml__estimator__min_samples_leaf": [5],
             "ml__estimator__max_depth": [5],
             "ml__estimator__random_state": [93],
             "ml__estimator__max_iter": [5]}

log_params = {}

enet_params = {"ml__estimator__C": [0.01],
               "ml__estimator__l1_ratio": [0.5]}

# smote ml grids
rf_params_sm = {"estimator__classifier__min_samples_split": [2],
             "estimator__classifier__max_depth": [5],
             "estimator__classifier__n_estimators": [10],
             "estimator__classifier__random_state": [93],
             "estimator__classifier__max_features": [0.3]}

gb_params_sm = {"estimator__classifier__min_samples_leaf": [2],
             "estimator__classifier__max_depth": [5],
             "estimator__classifier__random_state": [93],
             "estimator__classifier__max_iter": [10]}

log_params_sm = {"estimator__classifier__penalty": [None]}

enet_params_sm = {"estimator__classifier__C":[0.01], # Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization.
              "estimator__classifier__l1_ratio": [0.5],
              "estimator__classifier__max_iter": [10]}
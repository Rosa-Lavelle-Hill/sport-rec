nfolds = 2

rf_params = {'ml__estimator__max_depth': [5],
             'ml__estimator__max_features': [0.3],
             'ml__estimator__min_samples_split': [2],
             'ml__estimator__n_estimators': [50],
             'ml__estimator__random_state': [93]}

gb_params = {"ml__estimator__min_samples_leaf": [5],
             "ml__estimator__max_depth": [15],
             "ml__estimator__random_state": [93]}

log_params = {}

enet_params = {"ml__estimator__C": [0.01],
               "ml__estimator__l1_ratio": [0.5]}
nfolds = 5
rf_params = {"ml__estimator__min_samples_split": [2, 3, 4],
             "ml__estimator__max_depth": [15, 20],
             "ml__estimator__n_estimators": [300],
             "ml__estimator__random_state": [93],
             "ml__estimator__max_features": [0.6, 0.8, 1]}

gb_params = {"ml__estimator__min_samples_leaf": [5, 10, 20, 30],
             "ml__estimator__max_depth": [5, 10, 15, 20],
             "ml__estimator__random_state": [93]}

log_params = {'ml__estimator__C':[1, 10, 100, 1000],
              "ml__estimator__penalty": ['l2']}

enet_params = {"ml__estimator__C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],
              "ml__estimator__l1_ratio": [0.2, 0.5, 0.8]}
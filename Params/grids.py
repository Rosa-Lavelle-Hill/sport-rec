nfolds = 5
rf_params = {"classifier__min_samples_split": [2, 3, 4, 5],
             "classifier__max_depth": [10, 15, 20],
             "classifier__n_estimators": [300, 500, 700],
             "classifier__random_state": [93],
             "classifier__max_features": [0.3, 0.4, 0.5, 0.6]}

gb_params = {"classifier__min_samples_leaf": [5, 10, 20, 30],
             "classifier__max_depth": [5, 10, 15, 20],
             "classifier__random_state": [93]}

log_params = {'classifier__C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
              "classifier__penalty": ['l1', 'l2']}

enet_params = {"classifier__C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],
              "classifier__l1_ratio": [0.2, 0.5, 0.8]}
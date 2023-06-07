nfolds = 2

rf_params = {'classifier__max_depth': [15],
             'classifier__max_features': [0.3],
             'classifier__min_samples_split': [2],
             'classifier__n_estimators': [700],
             'classifier__random_state': [93]}

gb_params = {"classifier__min_samples_leaf": [5],
             "classifier__max_depth": [15],
             "classifier__random_state": [93]}

enet_params = {"classifier__tol": [0.001],
               "classifier__max_iter": [500],
               "classifier__l1_ratio": [0.5],
               "classifier__alpha": [0.5]}
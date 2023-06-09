nfolds = 2

rf_params = {'classifier__max_depth': [5],
             'classifier__max_features': [0.3],
             'classifier__min_samples_split': [2],
             'classifier__n_estimators': [50],
             'classifier__random_state': [93]}

gb_params = {"classifier__min_samples_leaf": [5],
             "classifier__max_depth": [15],
             "classifier__random_state": [93]}

log_params = {"classifier__C": [0.01],
              "classifier__penalty": ['l1']}

enet_params = {"classifier__C":[0.01],
              "classifier__l1_ratio": [0.5]}
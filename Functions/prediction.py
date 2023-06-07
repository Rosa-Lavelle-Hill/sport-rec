
import pandas as pd
import datetime as dt
import joblib

from fixed_params import decimal_places, scoring, verbose, random_state, nfolds, categorical_features
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from Functions.pipeline import construct_pipelines

def prediction(outcome, df,
               test_run,
               use_pre_trained,
               do_GB_only=False,
               do_testset_evaluation=True):

    # redefine X and y
    X = df.drop(outcome, axis=1)
    y = df[outcome]

    # construct pipes
    categorical_features.remove(outcome)
    numeric_features_index = X.drop(categorical_features, inplace=False, axis=1).columns
    categorical_features_index = X[categorical_features].columns
    pipe_rf, pipe_gb = construct_pipelines(numeric_features_index, categorical_features_index)
    pipes = [pipe_rf, pipe_gb]
    model_names = ["RF", "GB"]

    # split data into train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state,
                                                        test_size=0.2, shuffle=True, stratify=y)

    best_params_dict = {}
    params_save = "Results/Best_Params/"
    if use_pre_trained == False:
        if test_run == True:
            from Params.test_grids import rf_params, gb_params
        else:
            from Params.grids import rf_params, gb_params

        param_list = [rf_params, gb_params]

        # Training
        for model_name, pipe, params in zip(model_names, pipes, param_list):

            if do_GB_only == True:
                if (model_name == "LM") or (model_name == "RF"):
                    continue

            save_file = "Results/Prediction/{}.txt".format(model_name)

            print("Running {} model".format(model_name))
            print("{}:".format(model_name),
                  file=open(save_file, "w"))

            grid_search = GridSearchCV(estimator=pipe,
                                       param_grid=params,
                                       cv=nfolds,
                                       scoring=scoring,
                                       verbose=verbose,
                                       refit=False,
                                       n_jobs=2)

            # start timer
            grid_start = dt.datetime.now()
            # run the grid search
            grid_search.fit(X_train, y_train)
            # end timer
            grid_end = dt.datetime.now()
            training_time = grid_end - grid_start
            print("Training done. Time taken: {}".format(training_time), file=open(save_file, "a"))

            best_params = grid_search.best_params_
            joblib.dump(best_params, params_save + '{}.pkl'.format(model_name), compress=1)
            best_train_score = round(abs(grid_search.best_score_), decimal_places)
            print("params tried:\n{}\n".format(rf_params), file=open(save_file, "a"))

            print(
                "Best training {} score: {}. Best model params:\n{}.\n".format(scoring, best_train_score, best_params),
                file=open(save_file, "a"))

            best_params_dict[model_name] = best_params

    else:
        for model_name in model_names:
            best_params_dict[model_name] = joblib.load(params_save + '{}.pkl'.format(model_name))

    # Test Model

    test_scores = {}
    optimised_pipes = {}
    for model_name, best_model_params, pipe in zip(model_names, best_params_dict, pipes):

        if do_GB_only == True:
            if (model_name == "LM") or (model_name == "RF"):
                continue

        if model_name == "GB":
            continue
        #     todo: fix GB

        save_file = "Results/Prediction/{}.txt".format(model_name)
        best_model_param_values = best_params_dict[model_name]

        # set pipeline to use best params
        pipe.set_params(**best_model_param_values)
        optimised_pipes[model_name]=pipe

        # fit best pipeline to training data
        pipe.fit(X_train, y_train)

        # Test on Hold-out Data:
        if do_testset_evaluation == True:
            print("Evaluating performance on test set for {}".format(model_name))

            # use pipeline to make predictions
            y_pred = pipe.predict(X_test)

            # evaluate/score best out of sample
            test_score_f1_weighted = round(metrics.r2_score(y_test, y_pred), decimal_places)

            print("Best {} model performance on test data:\nR2: {}; mae: {}".format(model_name, test_score_f1_weighted),
                  file=open(save_file, "a"))

            test_scores[model_name] = {"F1_weighted": test_score_f1_weighted}

    if do_testset_evaluation == True:
        test_scores = pd.DataFrame.from_dict(test_scores)
        test_scores.to_csv(
            "Results/Prediction/all_test_scores.csv")

    return optimised_pipes



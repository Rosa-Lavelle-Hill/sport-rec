import numpy as np
import pandas as pd
import datetime as dt
import joblib
from imblearn.over_sampling import SMOTENC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

from Functions.plotting import plot_confusion_matrix
from fixed_params import decimal_places, single_label_scoring, multi_label_scoring, verbose, random_state, nfolds,\
    categorical_features, do_Enet, do_GB
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from Functions.pipeline import construct_pipelines, construct_smote_pipelines, construct_dummy_pipelines


def prediction(outcome, df,
               test_run, start_string, t,
               use_pre_trained,
               smote,
               multi_label,
               do_testset_evaluation,
               do_Enet=do_Enet,
               do_GB_only=False,
               ):

    # redefine X and y
    X = df.drop(outcome, axis=1)
    y = df[outcome]

    # transform X and y
    if multi_label == True:
        df.sort_values("ID_new", inplace=True)
        # transform y
        Y_i = df[['ID_new', outcome]].sort_values(by="ID_new")
        y = Y_i.groupby('ID_new')[outcome].apply(list)

        # select only unique X (join on grouped index)
        X.sort_values(by="ID_new", inplace=True)
        X = X.drop_duplicates(subset="ID_new")
        X.set_index("ID_new", drop=True, inplace=True)
        X_and_y = X.join(y)

        # redefine X and y:
        y = X_and_y[outcome]
        mlb = MultiLabelBinarizer()
        # todo: should be done on train and test set separately?****
        mlb.fit(y)
        y = mlb.transform(y)
        X = X_and_y.drop(outcome, axis=1)

    else:
        # remove person identifier
        X.drop('ID_new', axis=1, inplace=True)

    # construct pipes
    numeric_features_index = X.drop(categorical_features, inplace=False, axis=1).columns
    categorical_features_index = X[categorical_features].columns

    if smote == True:
        pipe_log, pipe_enet, pipe_rf, pipe_gb = construct_smote_pipelines(numeric_features_index, categorical_features_index, multi_label)
    else:
        pipe_log, pipe_enet, pipe_rf, pipe_gb = construct_pipelines(numeric_features_index, categorical_features_index, multi_label)

    pipes = [pipe_log, pipe_enet, pipe_rf, pipe_gb]
    model_names = ["Log", "Enet", "RF", "GB"]

    if do_GB == False:
        pipes.remove(pipe_gb)
        model_names.remove('GB')
    if do_Enet == False:
        pipes.remove(pipe_enet)
        model_names.remove('Enet')

    # split data into train and test splits (can only stratify with single label)
    if multi_label == True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state,
                                                            test_size=0.2, shuffle=True)
        if smote == True:
            # Find indices of categorical features
            categorical_features_indices = [X_train.columns.get_loc(col_name) for col_name in
                                            categorical_features]

            # Apply SMOTENC to training data
            smote = SMOTENC(sampling_strategy='auto', categorical_features=categorical_features_indices, random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state,
                                                            test_size=0.2, shuffle=True, stratify=y)

    best_params_dict = {}
    params_save = "Results/Best_Params/"
    if use_pre_trained == False:
        if test_run == True:
            if multi_label == True:
                from Params.multi_label.test_grids import log_params, enet_params, rf_params, gb_params
            else:
                from Params.test_grids import log_params, enet_params, rf_params, gb_params
        else:
            if multi_label == True:
                from Params.multi_label.grids import log_params, enet_params, rf_params, gb_params
            else:
                from Params.grids import log_params, enet_params, rf_params, gb_params


        param_list = [log_params, enet_params, rf_params, gb_params]

        # Train ML Models =========================================================================================
        for model_name, pipe, params in zip(model_names, pipes, param_list):
            # todo: uneven list lengths?
            if do_GB_only == True:
                if (model_name == "LM") or (model_name == "RF"):
                    continue

            save_file = "Results/Prediction/{}_{}{}.txt".format(model_name, start_string, t)

            print("Running {} model".format(model_name))
            print("{}:".format(model_name),
                  file=open(save_file, "w"))

            if multi_label == True:
                kfold = KFold(n_splits=nfolds, random_state=random_state, shuffle=True)
                scoring = multi_label_scoring
            else:
                # preserve the distribution of data across outcome classes
                scoring = single_label_scoring
                kfold = StratifiedKFold(n_splits=nfolds,
                                                   shuffle=True,
                                                   random_state=random_state)

            grid_search = GridSearchCV(estimator=pipe,
                                       param_grid=params,
                                       cv=kfold,
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
            joblib.dump(best_params, params_save + '{}_{}{}{}.pkl'.format(outcome, model_name, start_string, t), compress=1)
            best_train_score = round(abs(grid_search.best_score_), decimal_places)
            print("params tried:\n{}\n".format(rf_params), file=open(save_file, "a"))

            print(
                "Best training {} score: {}. Best model params:\n{}.\n".format(scoring, best_train_score, best_params),
                file=open(save_file, "a"))

            best_params_dict[model_name] = best_params

    else:
        for model_name in model_names:
            best_params_dict[model_name] = joblib.load(params_save + '{}_{}{}{}.pkl'.format(outcome, model_name, start_string, t))


    # Test Baseline Models =========================================================================================
    if do_testset_evaluation == True:

        test_scores = {}
        print("Evaluating performance on test set of baseline models")

        pipe_dum_mf, pipe_dum_zero, pipe_dum_random, pipe_dum_strat = construct_dummy_pipelines(numeric_features_index,
                                                                                                categorical_features_index,
                                                                                                multi_label)

        for dum_model, dum_name in zip([pipe_dum_mf, pipe_dum_zero, pipe_dum_random, pipe_dum_strat],
                                       ["Dummy_MF", "Dummy_Zero", "Dummy_Random", "Dummy_Stratified"]):
            dum_model.fit(X_train, y_train)
            y_pred = dum_model.predict(X_test)
            y_prob = dum_model.predict_proba(X_test)
            test_score_f1_weighted = round(metrics.f1_score(y_test, y_pred, average="weighted"), decimal_places)
            if multi_label == False:
                test_log_loss = round(metrics.log_loss(y_test, y_prob), decimal_places)

                test_scores[dum_name] = {"F1_weighted": test_score_f1_weighted,
                                    "Log_loss": test_log_loss}
            else:
                test_scores[dum_name] = {"F1_weighted": test_score_f1_weighted}
                dummy_results_dict = classification_report(
                    y_test,
                    y_pred,
                    output_dict=True,
                    zero_division=0.0
                )
                test_scores[dum_name] = {"micro_precision": round(dummy_results_dict['micro avg']['precision'], 2),
                                         "micro_f1": round(dummy_results_dict['micro avg']['f1-score'], 2),
                                         "weighted_precision": round(dummy_results_dict['weighted avg']['precision'], 2),
                                         "weighted_f1": round(dummy_results_dict['weighted avg']['f1-score'], 2)}

    # Test ML Models =========================================================================================
    optimised_pipes = {}
    for model_name, best_model_params, pipe in zip(model_names, best_params_dict, pipes):

        if do_GB_only == True:
            if (model_name != "GB"):
                continue

        save_file = "Results/Prediction/{}_{}{}.txt".format(model_name, start_string, t)
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
            y_prob = pipe.predict_proba(X_test)

            # evaluate/score best out of sample
            test_score_f1_weighted = round(metrics.f1_score(y_test, y_pred, average="weighted"), decimal_places)
            if multi_label == False:
                test_log_loss = round(metrics.log_loss(y_test, y_prob), decimal_places)
                test_cm = metrics.multilabel_confusion_matrix(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                print("Best {} model performance on test data;"
                      "F1_weighted: {}, log loss: {}, CM: \n {}".format(model_name, test_score_f1_weighted,
                                                                        test_log_loss, test_cm),
                      file=open(save_file, "a"))

                # plot confusion_matrix
                cm_save_path = "Results/Prediction/Confusion_Matrix/"
                plot_confusion_matrix(cm,
                                      target_names = range(1, 15),
                                      title='Confusion matrix',
                                      cmap=None,
                                      normalize=False,
                                      save_name="cm_{}_{}{}{}".format(model_name, outcome, start_string, t),
                                      save_path=cm_save_path)

                test_scores[model_name] = {"F1_weighted": test_score_f1_weighted,
                                           "Log_loss": test_log_loss}
            if multi_label == True:
                results_dict = classification_report(
                    y_test,
                    y_pred,
                    output_dict=True,
                    zero_division=0.0
                )
                print("Best {} model performance on test data:\n".format(model_name) +
                      str(classification_report(y_test, y_pred, output_dict=False)), file=open(save_file, "a"))

                m_cm = multilabel_confusion_matrix(y_test, y_pred)
                print("Confusion matrices:\n".format(model_name) +
                      " Predicted: \n"
                      "   N | P  \n"
                      "[[TN, FP] \n" +
                       "[FN, TP]] \n \n" +
                      str(m_cm), file=open(save_file, "a"))

                test_scores[model_name] = {"F1_weighted": test_score_f1_weighted}
                test_scores[model_name] = {"micro_precision": round(results_dict['micro avg']['precision'],2),
                                         "micro_f1": round(results_dict['micro avg']['f1-score'], 2),
                                         "weighted_precision": round(results_dict['weighted avg']['precision'], 2),
                                         "weighted_f1": round(results_dict['weighted avg']['f1-score'], 2)}

    if do_testset_evaluation == True:
        test_scores = pd.DataFrame.from_dict(test_scores)
        test_scores.to_csv(
            "Results/Prediction/all_test_scores_{}{}{}.csv".format(outcome, start_string, t))

    return optimised_pipes


# todo: "Invalid parameter ml for estimator"
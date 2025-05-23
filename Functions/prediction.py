import json

import numpy as np
import pandas as pd
import datetime as dt
import joblib
from joblib import dump, load
from imblearn.over_sampling import SMOTENC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from Functions.plotting import plot_confusion_matrix
from Functions.read import append_if_not_exists
from fixed_params import decimal_places, single_label_scoring, multi_label_scoring, verbose, random_state, nfolds,\
    categorical_features, test_size
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold, GroupShuffleSplit
from Functions.pipeline import construct_pipelines, construct_smote_pipelines, construct_dummy_pipelines


def prediction(outcome,
               df,
               test_run,
               start_string,
               t,
               use_pre_trained,
               smote,
               multi_label,
               do_testset_evaluation, # only runs if not using a pre-optimised model, otherwise loads results df
               predict_probab,
               do_Enet,
               do_GB,
               do_GB_only=False,
               dump_fitted_model=True,
               load_fitted_model=True
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

    if do_GB_only == True:
        model_names = ['GB']
        pipes = [pipe_gb]

    # split data into train and test splits (can only stratify with single label)... each indiv either in train or test (as y is grouped outcomes; i.e., 1 row per indiv)
    if multi_label == True:
        print(f"Constructing multi-label classification pipeline")

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state,
                                                            test_size=test_size, shuffle=True)

        # redefine y as 1 of K:
        mlb = MultiLabelBinarizer()
        y_train = mlb.fit_transform(y_train)
        y_test = mlb.transform(y_test)

        # save modelling data
        print(f"train data: {X_train.shape}; test data: {X_test.shape} ... where each row is an individual (y is multidimensional)")
        X_train.to_csv("Data/Modelling/MultiLab/NoSMOTE/X_train.csv", index=False)
        X_test.to_csv("Data/Modelling/MultiLab/NoSMOTE/X_test.csv", index=False)
        y_train_df = pd.DataFrame(y_train)
        y_test_df = pd.DataFrame(y_train)
        y_train_df.to_csv("Data/Modelling/MultiLab/NoSMOTE/y_train.csv", index=False)
        y_test_df.to_csv("Data/Modelling/MultiLab/NoSMOTE/y_test.csv", index=False)

        # if smote == True:
        #     # Find indices of categorical features
        #     categorical_features_indices = [X_train.columns.get_loc(col_name) for col_name in
        #                                     categorical_features]
        #
        #     # Apply SMOTENC to training data
        #     smote = SMOTENC(sampling_strategy='auto', categorical_features=categorical_features_indices, random_state=42)
        #     X_train, y_train = smote.fit_resample(X_train, y_train)

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state,
                                                            test_size=test_size, shuffle=True, stratify=y)
    #     todo: ***if run, need to split based on indiv

    best_params_dict = {}
    params_save = "Results/Best_Params/"
    if use_pre_trained == False:
        # test params:
        if test_run == True:
            if multi_label == True:
                if smote == True:
                    from Params.multi_label.test_grids import (log_params_sm as log_params,
                                                               enet_params_sm as enet_params,
                                                               rf_params_sm as rf_params,
                                                               gb_params_sm as gb_params)
                if smote == False:
                    from Params.multi_label.test_grids import log_params, enet_params, rf_params, gb_params
            if multi_label == False:
                from Params.test_grids import log_params, enet_params, rf_params, gb_params
        # actual params:
        if test_run == False:
            if multi_label == True:
                if smote == True:
                    from Params.multi_label.grids import (log_params_sm as log_params,
                                                            enet_params_sm as enet_params,
                                                            rf_params_sm as rf_params,
                                                            gb_params_sm as gb_params)
                if smote == False:
                    from Params.multi_label.grids import log_params, enet_params, rf_params, gb_params
            if multi_label == False:
                from Params.grids import log_params, enet_params, rf_params, gb_params

        param_list = [log_params, enet_params, rf_params, gb_params]

        if do_GB == False:
            param_list.remove(gb_params)
        if do_Enet == False:
            param_list.remove(enet_params)
        if do_GB_only == True:
            param_list = [gb_params]

        # Train ML Models =========================================================================================
        for model_name, pipe, params in zip(model_names, pipes, param_list):
            if do_GB_only == True:
                if (model_name == "LM") or (model_name == "RF"):
                    continue

            save_file = "Results/Prediction/{}_{}{}.txt".format(model_name, start_string, t)

            print("Running {} model".format(model_name))
            model_start_time = dt.datetime.now()
            print("{}:".format(model_name),
                  file=open(save_file, "w"))

            if multi_label == True:
                kfold = KFold(n_splits=nfolds, random_state=random_state, shuffle=True)
                scoring = multi_label_scoring
            else:
                # preserve the distribution of data across outcome classes
                scoring = single_label_scoring
                kfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=random_state)

            grid_search = GridSearchCV(estimator=pipe,
                                       param_grid=params,
                                       cv=kfold,
                                       scoring=scoring,
                                       verbose=verbose,
                                       refit=False,
                                       n_jobs=2,
                                       error_score="raise")

            # count distribution of class labels within each fold:
            if model_name == "Log":
                for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
                    y_train_fold = y_train[train_idx]
                    print(f"\nFold {fold + 1}:")
                    for label_idx in range(y_train.shape[1]):
                        label = f"Label_{label_idx}"
                        count = np.sum(y_train_fold[:, label_idx])
                        total = y_train_fold.shape[0]
                        proportion = (count / total) * 100
                        print(f"  {label}: {count}/{total} positives ({proportion:.2f}%)")

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

            # dump fitted model
            if dump_fitted_model == True:
                pipe.set_params(**best_params)
                pipe.fit(X_train, y_train)
                model_save = "Results/Fitted_Models/"
                joblib.dump(pipe, f'{model_save}{outcome}_{model_name}{start_string}{t}.joblib', compress=1)

            best_train_score = round(abs(grid_search.best_score_), decimal_places)
            print("params tried:\n{}\n".format(params), file=open(save_file, "a"))

            print(
                "Best training {} score: {}. Best model params:\n{}.\n".format(scoring, best_train_score, best_params),
                file=open(save_file, "a"))

            best_params_dict[model_name] = best_params

            # runtime
            model_end_time = dt.datetime.now()
            model_run_time = model_end_time - model_start_time
            print(f'{model_name} finished training. Run time: {model_run_time}')

    else:
        for model_name in model_names:
            best_params_dict[model_name] = joblib.load(params_save + '{}_{}{}{}.pkl'.format(outcome, model_name, start_string, t))


    # Test Baseline Models =========================================================================================
    if (do_testset_evaluation == True):

        test_scores = {}
        all_base_results = {}
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
                test_scores[dum_name] = {"micro_precision": round(dummy_results_dict['micro avg']['precision'], 3),
                                         "micro_f1": round(dummy_results_dict['micro avg']['f1-score'], 3),
                                         "weighted_precision": round(dummy_results_dict['weighted avg']['precision'], 3),
                                         "weighted_f1": round(dummy_results_dict['weighted avg']['f1-score'], 3 )}
                all_base_results[dum_name] = dummy_results_dict

        # save baselines per category to plot later
        all_base_results_df = pd.DataFrame.from_dict(all_base_results)
        all_base_results_df.to_csv("Results/Prediction/Baseline_per_category/all_base_scores_{}{}{}.csv".format(outcome, start_string, t))
        with open("Results/Prediction/Baseline_per_category/all_base_scores_{}{}{}.json".format(outcome, start_string, t), "w") as json_file:
            json.dump(all_base_results, json_file, indent=4)

        # Test ML Models =========================================================================================

        optimised_pipes = {}
        for model_name, best_model_params, pipe in zip(model_names, best_params_dict, pipes):
            print("Evaluating performance on test set for {}".format(model_name))

            if do_GB_only == True:
                if (model_name != "GB"):
                    continue

            save_file = "Results/Prediction/{}_{}{}.txt".format(model_name, start_string, t)

            if load_fitted_model == False:
                best_model_param_values = best_params_dict[model_name]

                # set pipeline to use best params
                pipe.set_params(**best_model_param_values)
                optimised_pipes[model_name]=pipe

                # fit best pipeline to training data
                pipe.fit(X_train, y_train)

                # dump fitted model
                if dump_fitted_model == True:
                    model_save = "Results/Fitted_Models/"
                    joblib.dump(pipe, f'{model_save}{outcome}_{model_name}{start_string}{t}.joblib', compress=1)

            if load_fitted_model == True:
                pipe = load(f'Results/Fitted_Models/{outcome}_{model_name}{start_string}{t}.joblib')
                optimised_pipes[model_name] = pipe
            #     ^ technically a fitted pipe

            # use pipeline to make predictions
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)

            # evaluate/score best out of sample
            test_score_f1_weighted = round(metrics.f1_score(y_test, y_pred, average="weighted"), decimal_places)
            if multi_label == False:
                test_log_loss = round(metrics.log_loss(y_test, y_prob), decimal_places)
                test_cm = metrics.multilabel_confusion_matrix(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                append_if_not_exists(save_file, "Best {} model performance on test data;"
                      "F1_weighted: {}, log loss: {}, CM: \n {}".format(model_name, test_score_f1_weighted,
                                                                        test_log_loss, test_cm))

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
                append_if_not_exists(save_file, "Best {} model performance on test data:\n".format(model_name) +
                      str(classification_report(y_test, y_pred, output_dict=False)))

                m_cm = multilabel_confusion_matrix(y_test, y_pred)

                append_if_not_exists(save_file, "Confusion matrices:\n".format(model_name) +
                      " Predicted: \n"
                      "   N | P  \n"
                      "[[TN, FP] \n" +
                       "[FN, TP]] \n \n" +
                      str(m_cm))

                test_scores[model_name] = {"micro_precision": round(results_dict['micro avg']['precision'], 3),
                                         "micro_f1": round(results_dict['micro avg']['f1-score'], 3),
                                         "weighted_precision": round(results_dict['weighted avg']['precision'], 3),
                                         "weighted_f1": round(results_dict['weighted avg']['f1-score'], 3)}

        if predict_probab == True:
            if multi_label == True:
                rec_save_path = "Results/Recommendations/"
                print('predicting probabilities')
                with open("Data/Dicts_and_Lists/short_names_dict.json", 'r') as file:
                    short_names_dict = json.load(file)
                predicted_probabilities = pipe.predict_proba(X_test)
                class_names_orig = mlb.classes_
                short_names_dict = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in short_names_dict.items()}
                positive_label_probabilities = [probs[:, 1] for probs in predicted_probabilities]
                df_predicted_probabilities = np.transpose(pd.DataFrame(positive_label_probabilities))
                df_predicted_probabilities.columns = list(short_names_dict.values())
                df_predicted_probabilities = round(df_predicted_probabilities, 2)
                df_predicted_probabilities.to_excel(rec_save_path + "all_probabilities_{}_{}{}.xlsx".format(model_name, start_string, t))
                # Calculate class rankings
                class_rankings = np.argsort(-df_predicted_probabilities.values, axis=1) + 1
                # Create a DataFrame of class rankings
                df_class_rankings = pd.DataFrame(class_rankings,
                                                 columns=[f'Rank_{i}' for i in range(1, len(class_rankings[0]) + 1)])

                df_class_rankings.to_excel(rec_save_path + "all_recomendations_{}_{}{}.xlsx".format(model_name, start_string, t))
                # Convert to top K recommendations
                K=3
                df_class_rankings_K = df_class_rankings.iloc[:, 0:K]
                df_class_rankings_K.to_excel(
                    rec_save_path + "{}_recomendations_{}_{}{}.xlsx".format(K, model_name, start_string, t))

        # save data for plotting
        test_scores = pd.DataFrame.from_dict(test_scores)
        test_scores.to_csv("Results/Prediction/all_test_scores_{}{}{}.csv".format(outcome, start_string, t))

    return optimised_pipes, model_names


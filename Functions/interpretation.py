import pandas as pd
import shap
import json
import pickle
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from Functions.plotting import plot_impurity, plot_permutation, plot_SHAP, plot_impurity_ml, heatmap_importance, \
    plot_SHAP_df, plot_SHAP_force, plot_forceSHAP_df
from fixed_params import decimal_places, multi_label_scoring, single_label_scoring, verbose,\
    random_state, categorical_features, multi_label
from fixed_params import n_shap_features as n_features
from Functions.preprocessing import remove_cols, get_preprocessed_col_names
from sklearn.preprocessing import MultiLabelBinarizer

def interpretation(df,
                   outcome,
                   optimised_pipes,
                   start_string,
                   t,
                   do_impurity_importance,
                   do_permutation_importance,
                   do_SHAP_importance,
                   recalc_SHAP,
                   model_names,
                   best_model
                   ):

    # redefine X and y
    X = df.drop(outcome, axis=1)
    y = df[outcome]

    # transform X and y
    # todo: save transformed data and load from file
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

    else:
        # remove person identifier
        X.drop('ID_new', axis=1, inplace=True)


    # split data into train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state,
                                                        test_size=0.2, shuffle=True)

    # redefine y as 1 of K:
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    y_test = mlb.transform(y_test)

    if best_model:
        model_names = [best_model]

    for model_name in model_names:

        pipe = optimised_pipes[model_name]
        feature_names = get_preprocessed_col_names(X, pipe, cat_vars=categorical_features)
        with open("Data/Dicts_and_Lists/column_names.json", 'r') as file:
            feature_names_dict = json.load(file)
        nice_feature_names = [feature_names_dict[key] for key in feature_names]

        # Permutation Importance (across all categories):
        if do_permutation_importance == True:

            print('starting permutation importance for model {}...'.format(model_name))
            pipe.fit(X_train, y_train)
            result = permutation_importance(pipe, X_test, y_test, n_repeats=1, random_state=93, n_jobs=2)

            perm_importances = result.importances_mean
            vars = X.columns.to_list()
            nice_var_names = [feature_names_dict[key] for key in vars]
            dict = {'Feature': nice_var_names, "Importance": perm_importances}
            perm_imp_df = pd.DataFrame(dict)
            # just get most important x
            perm_imp_df.sort_values(by="Importance", ascending=False, inplace=True, axis=0)
            perm_imp_df = perm_imp_df
            # flip so most important at top on graph
            perm_imp_df.sort_values(by="Importance", ascending=True, inplace=True, axis=0)
            perm_imp_df.to_csv("Results/Importance/Permutation/{}_{}{}.csv".format(model_name,
                                                                                        start_string, t))

            # plot
            plot_permutation(perm_imp_df=perm_imp_df,
                          save_path="Results/Importance/Permutation/Plots/",
                          save_name="{}_perm_{}{}".format(model_name, start_string, t))

        if (do_impurity_importance == True) or (do_SHAP_importance == True):
            # fit to and transform train
            preprocessor = pipe.named_steps['preprocessor']
            opt_model = pipe.named_steps.ml
            X_train_p = preprocessor.fit_transform(X_train)
            opt_model.fit(X_train_p, y_train)
            # transform test
            X_test_p = preprocessor.transform(X_test)
            opt_model.fit(X_test_p, y_test)

        # Impurity-based Importance:
        if do_impurity_importance == True:
            if (model_name == "RF"):

                # Get the list of Random Forest estimators from MultiOutputClassifier
                estimators_list = pipe.named_steps['ml'].estimators_

                # Initialize a list to store the feature importances for each output
                feature_importances_per_output = []

                # Loop through each estimator (output) and retrieve feature importances
                for estimator in estimators_list:
                    feature_importances_per_output.append(estimator.feature_importances_)

                # Convert the list to a numpy array for easier handling
                feature_importances_array = np.array(feature_importances_per_output)

                # Convert the feature importances array to a DataFrame
                impurity_imp_df = pd.DataFrame(
                    feature_importances_array,
                    columns=feature_names,
                    index=[f'Category_{i}' for i in range(1, len(estimators_list) + 1)])

                # plot mean impurity importance
                plot_impurity_ml(impurity_imp_df, save_path = "Results/Importance/Impurity/Plots/",
                              save_name = "{}_impurity_{}{}".format(model_name, start_string, t))

                # plot per sport classification using a heatmap:
                heatmap_importance(df=impurity_imp_df,
                                   title="", xlab="Sports",
                                   ylab="Feature", save_name="Heatmap_RF_{}".format(start_string),
                                   save_path="Results/Importance/Impurity/Plots/")

        if do_SHAP_importance == True:
            filename = f"Results/Importance/SHAP/{outcome}/SHAP_imp_{model_name}_{start_string}{t}.pkl"
            shap_plot_save_path = f"Results/Importance/SHAP/{outcome}/Plots/"
            # for now, only do SHAP for RF
            if (model_name == "RF") or (model_name == "GB"):
                if recalc_SHAP == True:
                    print('starting SHAP importance for model {}...'.format(model_name))

                    # Create a DataFrame to store SHAP values
                    shap_df = pd.DataFrame(columns=['Class', 'Label', 'Feature', 'SHAP Value'])

                    # Loop through each class's estimator
                    shap_dict = {}
                    for cat_num in range(len(df[outcome].unique())):
                        # Calculate SHAP values for the class-label combination
                        explainer = shap.TreeExplainer(opt_model.estimators_[cat_num])
                        shap_values = explainer.shap_values(X_test_p, check_additivity=True)

                        # extract SHAP values for class 1 only
                        if shap_values.ndim == 3:
                            shap_values_class1 = shap_values[:, :, 1]
                        if shap_values.ndim == 2:
                            shap_values_class1 = shap_values

                        # Create a DataFrame from the SHAP values for class 1
                        class_shap_df = pd.DataFrame(shap_values_class1, columns=feature_names)

                        # Save to enable reload
                        save_name = f"category_{cat_num}_{model_name}.csv"
                        class_shap_df.to_csv(f"Results/Importance/SHAP/{outcome}/"+save_name)
                        shap_dict[cat_num] = class_shap_df

                        # # Demo individual shap plots for a few individuals
                        # plot_type = "force"
                        # # todo: calculate base value as mean of test set predictions/actual ratio of 0:1s
                        # for i in range(1, 3):
                        #     save_name = f"SHAP{start_string}_{model_name}_{plot_type}_category_{cat_num}_person{i}_{t}"
                        #     base_val = 0.5
                        #     cat_shap_df = shap_dict[cat_num]
                        #     plot_forceSHAP_df(cat_shap_df, explainer, i, base_val, col_list=feature_names,
                        #                       save_path=shap_plot_save_path,
                        #                       save_name=save_name, title=f"Category {cat_num + 1}")

                    # Save the dictionary as a pickle
                    with open(filename, 'wb') as file:
                        pickle.dump(shap_dict, file)

                if recalc_SHAP == False:
                    # load SHAP dict from file
                    with open(filename, 'rb') as file:
                        shap_dict = pickle.load(file)

                # Get SHAP plot for each category separately:
                X_test_p_df = pd.DataFrame(X_test_p, columns=feature_names)
                with open("Data/Dicts_and_Lists/short_names_dict.json", 'r') as file:
                    short_names_dict = json.load(file)
                # for each outcome category
                for cat_num in list(short_names_dict.keys()):
                    cat_name = short_names_dict[cat_num]
                    cat_shap_df = shap_dict[int(cat_num)-1]
                    # get shap values:
                    shap_array = cat_shap_df.values

                    plot_type = "bar"
                    save_name = f"SHAP{start_string}_{model_name}_{plot_type}_category_{cat_num}{t}"
                    plot_SHAP_df(shap_array, X_df=X_test_p_df, col_list=nice_feature_names,
                              n_features=n_features, plot_type=plot_type,
                              save_path=shap_plot_save_path,
                              save_name=save_name, title=f"{cat_name}")

                    plot_type = "summary"
                    save_name = f"SHAP{start_string}_{model_name}_{plot_type}_category_{cat_num}{t}"
                    plot_SHAP_df(shap_array, X_df=X_test_p_df, col_list=nice_feature_names,
                              n_features=n_features, plot_type=None,
                              save_path=shap_plot_save_path,
                              save_name=save_name, title=f"{cat_name}")

                    plot_type = "violin"
                    save_name = f"SHAP{start_string}_{model_name}_{plot_type}_category_{cat_num}{t}"
                    plot_SHAP_df(shap_array, X_df=X_test_p_df, col_list=nice_feature_names,
                              n_features=n_features, plot_type=plot_type,
                              save_path=shap_plot_save_path,
                              save_name=save_name, title=f"{cat_name}")


                    # # if interaction_plots == True:
                    #     if only_df1 == True:
                    #         if df_num != '1':
                    #             continue
                    #     SHAP_tree_interaction(X=X, y=y, df_num=df_num, start_string=start_string,
                    #                           rf_params=rf_params, names=names, n_inter_features=10,
                    #                           save_path=analysis_path + "Results/Importance/Plots/Test_Data/{}/interaction/".format(
                    #                               model))

                    # todo: treat each shap df as different? currently 10x importance dfs



    return
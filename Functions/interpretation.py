import pandas as pd
import shap
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from Functions.plotting import plot_impurity, plot_permutation, plot_SHAP, plot_impurity_ml
from fixed_params import decimal_places, multi_label_scoring, single_label_scoring, verbose,\
    random_state, categorical_features, multi_label
from Functions.preprocessing import remove_cols
from sklearn.preprocessing import MultiLabelBinarizer

def interpretation(df, outcome, optimised_pipes,
                   start_string, t,
                   do_impurity_importance,
                   do_permutation_importance,
                   do_SHAP_importance
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

        # redefine X and y:
        y = X_and_y[outcome]
        mlb = MultiLabelBinarizer()
        mlb.fit(y)
        y = mlb.transform(y)
        X = X_and_y.drop(outcome, axis=1)

    else:
        # remove person identifier
        X.drop('ID_new', axis=1, inplace=True)


    # split data into train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state,
                                                        test_size=0.2, shuffle=True)

    model_names = ["Log", "RF"]
    for model_name in model_names:

        pipe = optimised_pipes[model_name]

        # Impurity-based Importance:
        if do_impurity_importance == True:
            if (model_name == "RF"):

                # fit to and transform train
                preprocessor = pipe.named_steps['preprocessor']
                opt_model = pipe.named_steps.ml
                X_train_p = preprocessor.fit_transform(X_train)
                opt_model.fit(X_train_p, y_train)
                # transform test
                X_test_p = preprocessor.transform(X_test)

                opt_model.fit(X_test_p, y_test)
                # todo: need to transform data********

                # Get the list of Random Forest estimators from MultiOutputClassifier
                estimators_list = pipe.named_steps['ml'].estimators_

                # Initialize a list to store the feature importances for each output
                feature_importances_per_output = []

                # Loop through each estimator (output) and retrieve feature importances
                for estimator in estimators_list:
                    feature_importances_per_output.append(estimator.feature_importances_)

                # Convert the list to a numpy array for easier handling
                feature_importances_array = np.array(feature_importances_per_output)

                feature_names = get_preprocessed_col_names(X, pipe, cat_vars=categorical_features)

                # Convert the feature importances array to a DataFrame
                impurity_imp_df = pd.DataFrame(
                    feature_importances_array,
                    columns=feature_names,
                    index=[f'Category_{i}' for i in range(1, len(estimators_list) + 1)])

                # plot mean impurity importance
                plot_impurity_ml(impurity_imp_df, save_path = "Results/Importance/Impurity/Plots/",
                              save_name = "{}_impurity_{}{}".format(model_name, start_string, t))

                # plot per sport classification using a heatmap:





        # Permutation Importance:
        if do_permutation_importance == True:

            print('starting permutation importance for model {}...'.format(model_name))
            pipe.fit(X_train, y_train)
            result = permutation_importance(pipe, X_test, y_test, n_repeats=1, random_state=93, n_jobs=2)

            perm_importances = result.importances_mean
            vars = X.columns.to_list()
            dict = {'Feature': vars, "Importance": perm_importances}
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

        if do_SHAP_importance == True:
            if model_name != 'Log':
                print('starting SHAP importance for model {}...'.format(model_name))

                # Fit the explainer
                opt_model = pipe.named_steps.ml
                opt_model.fit(X_test, y_test)
                def shap_predict(inputs):
                    return opt_model.predict_proba(inputs)
                explainer = shap.Explainer(shap_predict, X_test)

                # Calculate the SHAP values and save
                shap_values = explainer(X_test[0])
                names = list(X.columns.values)
                shap_values_df = pd.DataFrame(shap_values, columns=names)
                shap_values_df.to_csv("Results/Importance/SHAP/SHAP_imp_{}_{}{}.csv".format(model_name,
                                                                                             start_string, t))

                n_features = 10
                shap_plot_save_path = "Results/Importance/SHAP/Plots/"

                shap_values = explainer(X_test)

                plot_type = "bar"
                plot_SHAP(shap_values, col_list=names,
                          n_features=n_features, plot_type=plot_type,
                          save_path=shap_plot_save_path,
                          save_name="{}_SHAP_{}_nfeatures{}_{}{}.png".format(model_name, plot_type, n_features,
                                                                             start_string, t))

                plot_type = "summary"
                plot_SHAP(shap_values, col_list=names,
                          n_features=n_features, plot_type=None,
                          save_path=shap_plot_save_path,
                          save_name="{}_SHAP_{}_nfeatures{}_{}{}.png".format(model_name, plot_type, n_features,
                                                                             start_string, t))

                plot_type = "violin"
                plot_SHAP(shap_values, col_list=names,
                          n_features=n_features, plot_type=plot_type,
                          save_path=shap_plot_save_path,
                          save_name="{}_SHAP_{}_nfeatures{}_{}{}.png".format(model_name, plot_type, n_features,
                                                                             start_string, t))

                # if interaction_plots == True:
                #     if only_df1 == True:
                #         if df_num != '1':
                #             continue
                #     SHAP_tree_interaction(X=X, y=y, df_num=df_num, start_string=start_string,
                #                           rf_params=rf_params, names=names, n_inter_features=10,
                #                           save_path=analysis_path + "Results/Importance/Plots/Test_Data/{}/interaction/".format(
                #                               model))

    return
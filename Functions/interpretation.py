import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from Functions.plotting import plot_impurity, plot_permutation, plot_SHAP
from fixed_params import decimal_places, multi_label_scoring, single_label_scoring, verbose,\
    random_state, categorical_features, multi_label
from preprocess import remove_cols
from sklearn.preprocessing import MultiLabelBinarizer

def interpretation(df, outcome, optimised_pipes,
                   start_string, t,
                   do_impurity_importance=True,
                   do_permutation_importance=True,
                   do_SHAP_importance=True
                   ):

    # redefine X and y
    X = df.drop(outcome, axis=1)
    y = df[outcome]
    display_n = len(X.columns) + 1

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

    model_names = ["LM", "RF"]
    for model_name in model_names:

        pipe = optimised_pipes[model_name]
        pipe.fit(X_train, y_train)

        # Impurity-based Importance:
        if do_impurity_importance == True:
            if (model_name == "RF"):

                feature_importances = pipe.named_steps["regressor"].feature_importances_

                vars = list(X.columns)
                dict = {'Feature': vars, "Importance": feature_importances}
                impurity_imp_df = pd.DataFrame(dict)

                # just get most important x
                impurity_imp_df.sort_values(by="Importance", ascending=False, inplace=True, axis=0)
                impurity_imp_df = impurity_imp_df[0:display_n]

                # flip so most important at top on graph
                impurity_imp_df.sort_values(by="Importance", ascending=True, inplace=True, axis=0)
                impurity_imp_df.to_csv("Results/Importance/Impurity/{}_{}{}.csv".format(model_name,
                                                                                             start_string, t))

                # plot
                plot_impurity(impurity_imp_df=impurity_imp_df, display_n=display_n,
                              save_path = "Results/Importance/Impurity/Plots/",
                              save_name = "{}_impurity_{}{}".format(model_name, start_string, t))

        # Permutation Importance:
        if do_permutation_importance == True:
            print('starting permutation importance for model {}...'.format(model_name))
            result = permutation_importance(pipe, X_test, y_test, n_repeats=1, random_state=93, n_jobs=2)

            perm_importances = result.importances_mean
            vars = X.columns.to_list()
            dict = {'Feature': vars, "Importance": perm_importances}
            perm_imp_df = pd.DataFrame(dict)
            # just get most important x
            perm_imp_df.sort_values(by="Importance", ascending=False, inplace=True, axis=0)
            perm_imp_df = perm_imp_df[0:display_n]
            # flip so most important at top on graph
            perm_imp_df.sort_values(by="Importance", ascending=True, inplace=True, axis=0)
            perm_imp_df.to_csv("Results/Importance/Permutation/{}_{}{}.csv".format(model_name,
                                                                                        start_string, t))

            # plot
            plot_permutation(perm_imp_df=perm_imp_df, display_n=display_n,
                          save_path="{}/Results{}/Importance/Permutation/Plots/".format(analysis, m),
                          save_name="{}_perm_{}{}".format(model_name, start_string, t))

        if do_SHAP_importance == True:
            if model_name != 'LM':
                print('starting SHAP importance for model {}...'.format(model_name))
                # Fit the explainer
                opt_model = pipe.named_steps.regressor
                explainer = shap.TreeExplainer(opt_model, feature_pertubation="tree_path_dependent")
                # Calculate the SHAP values and save
                shap_values = explainer.shap_values(X_test)
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
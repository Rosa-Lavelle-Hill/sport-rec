import numpy as np
import pandas as pd
from fixed_params import categorical_features, goal_vars
from Functions.plotting import plot_hist, plot_count


def preprocess(df, outcome):

    # drop rows were dv Missing:
    print(df.shape)
    df.dropna(subset=[outcome], axis=0, inplace=True, how='any')
    print(df.shape)

    # missing data inspection
    save_path = "Outputs/Descriptives/Missing/"

    missing_sum = pd.DataFrame(df.isna().sum())
    missing_perc = round((missing_sum / len(df)) * 100, 2)
    missing_summary = pd.concat([missing_sum, missing_perc], axis=1)
    missing_summary.reset_index(inplace=True)
    missing_summary.columns = ["Variable", "Missing_Sum", "Missing_Perc"]
    missing_summary.sort_values(by='Missing_Perc', ascending=False, inplace=True)
    missing_summary.to_csv(save_path + "iv_missing.csv")

    plot_hist(fig_size=(5, 5),
              bins=500,
              fontsize=12,
              x=missing_summary["Missing_Perc"],
              title="Missing Data Percentages for Variables",
              xlab="Missing Data Percentage (%)",
              ylab="Frequency",
              save_name="iv_missing_hist",
              save_path=save_path)

    # check missing data at indiv level
    missing_indiv_level_df = pd.DataFrame(df.isnull().sum(axis=1))
    missing_indiv_level_df.columns = ["Raw_missing_cols"]
    # calculate percentage
    missing_indiv_level_df["Percentage"] = round((missing_indiv_level_df["Raw_missing_cols"] / df.shape[1]) * 100, 4)
    # sort by percentage
    missing_indiv_level_df.sort_values(by="Percentage", ascending=False, inplace=True)
    missing_indiv_level_df.to_csv(save_path + "indiv_missing.csv")

    plot_hist(fig_size=(5, 5),
              bins=50,
              fontsize=12,
              x=missing_indiv_level_df["Percentage"],
              title="Missing data for individuals",
              xlab="Percentage",
              ylab="Frequency",
              save_name="indiv_missing_hist",
              save_path=save_path
              )

    # remove indivs where all goals missing:
    df.dropna(axis=0, how="all", subset=goal_vars, inplace=True)
    print(df.shape)

    # plot freq of categories after data dropped
    very_short_names = pd.read_csv("Data/Meta/v_short_outcome_names.csv", index_col=[0])
    very_short_names = list(very_short_names['0'])
    short_names = pd.read_csv("Data/Meta/short_outcome_names.csv", index_col=[0])
    short_names = list(short_names['0'])
    plot_count(data=df, x=outcome, hue=outcome, xlabs=very_short_names,
               save_path="Outputs/Descriptives/Modelling_df/", save_name="y_hist",
               xlab="Sport Category (B)", leg_labs=short_names, title="Distribution of outcome variable")

    # recode cat vars:
    for var in categorical_features:
        df[var] = pd.Categorical(df[var])

    # look at correlations:
    df_num = df.drop(categorical_features, axis=1)
    save_path= "Outputs/Descriptives/Correlations/"
    check_cors(df_num, save_path=save_path, save_name="correlations.csv")

    return df



def check_cors(X, save_path, save_name):
    corr_iv_m = X.corr(method='pearson', min_periods=100).abs()
    # drop repetitious pairs (diagonals and below in matrix):
    corr_iv = (corr_iv_m.where(np.triu(np.ones(corr_iv_m.shape), k=1).astype(np.bool))
               .stack()
               .sort_values(ascending=False))
    # save as a df
    corr_iv = pd.DataFrame(corr_iv)
    # rename col
    corr_iv.columns = ["Absolute_corr"]
    corr_iv.to_csv(save_path + save_name)
    return corr_iv
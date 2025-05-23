import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fixed_params import categorical_features, goal_vars, x_lab, person_id, answer_id
from Functions.plotting import plot_hist, plot_count, plot_by_var, plot_perc


def preprocess(df, outcome):
    # drop rows where dv Missing:
    print("original data size")
    print(df.shape)
    df.dropna(subset=[outcome], axis=0, inplace=True, how='any')
    print("dropped where no DV info")
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
    print("dropped where not all goals present")
    df.dropna(axis=0, how="all", subset=goal_vars, inplace=True)
    print(df.shape)

    # remove duplicate rows (not including index):
    print("dropped duplicate rows")
    columns_to_consider = [col for col in df.columns if col != "Index1"]
    df.drop_duplicates(inplace=True, subset=columns_to_consider)
    print(df.shape)

    # plot freq of categories after data dropped
    very_short_names = pd.read_csv("Data/Meta/v_short_outcome_names_{}.csv".format(outcome), index_col=[0])
    very_short_names = list(very_short_names['0'])

    with open('Data/Dicts_and_Lists/cat_nums.json', 'r') as file:
        cat_nums = json.load(file)

    with open('Data/Dicts_and_Lists/short_names_dict.json', 'r') as file:
        short_names_dict = json.load(file)

    # plot freq of categories
    df[outcome] = df[outcome].astype('category')
    plot_count(data=df, x=outcome, hue=outcome, xlabs=very_short_names,
               save_path="Outputs/Descriptives/Modelling_df/", save_name="y_hist_{}".format(outcome),
               xlab=x_lab, title="Distribution of outcome variable",
               order=cat_nums, leg_title=x_lab, label_dict=short_names_dict)

    # plot freq of categories by gender
    plot_by_var(data=df, x=outcome, hue="sex", xlabs=very_short_names,
                save_path="Outputs/Descriptives/Modelling_df/", save_name="y_hist_gender_{}".format(outcome),
                xlab=x_lab, leg_labs=["Women", "Men"],
                title="Distribution of outcome variable by gender")

    # plot % of categories
    plot_perc(data=df, x=outcome, hue=outcome, xlabs=very_short_names,
              save_path="Outputs/Descriptives/Modelling_df/", save_name="y_perc_{}".format(outcome),
              xlab=x_lab, title="Distribution of outcome variable (%)",
              order=cat_nums, leg_title=x_lab, label_dict=short_names_dict)

    # recode cat vars:
    for var in categorical_features:
        df[var] = pd.Categorical(df[var])
    df[outcome] = pd.Categorical(df[outcome])

    # look at correlations:
    df_num = df.drop(categorical_features, axis=1)
    save_path= "Outputs/Descriptives/Correlations/"
    check_cors(df_num, save_path=save_path, save_name="correlations.csv")

    # count unique individuals:
    unique_ids = df['ID_new'].nunique()
    print("unique individuals: " + str(unique_ids))

    # drop answer id
    df.drop([answer_id], axis=1, inplace=True)

    # plot the distributions of the Goal variables across...
    sns.set_style("whitegrid")
    save_path = "Outputs/Descriptives/Goal_Distributions/"
    with open('Data/Dicts_and_Lists/goal_col_names.json', 'r') as file:
        goal_col_names = json.load(file)
    label_dict = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in short_names_dict.items()}

    for var in goal_vars:
        var_name = goal_col_names[var]
        plt.figure(figsize=(10, 6))

        # 1) whole sample
        sns.kdeplot(df[var], fill=True, color="black", label="Overall", alpha=0.5)

        # 2) each category
        for category in df[outcome].unique():
            subset = df[df[outcome] == category]
            sns.kdeplot(subset[var], fill=True, label=f"{label_dict[category]}", alpha=0.5)

        # Formatting
        plt.title(f"Distribution of {var_name}", fontsize=14)
        plt.xlabel(var_name)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}{outcome}_{var}.png")
        plt.close()
        plt.clf()
        plt.cla()
        plt.close()

    # Get descriptives for paper:
    print("Additional descriptives for paper...")
    descriptives_file = "/Users/dhlab-admin/Documents/Rosa/sport-rec/Outputs/Descriptives/paper_descriptives_print.txt"
    
    # a) what % reported 2 categories
    counts = df['ID_new'].value_counts()
    total_unique = len(counts)
    percent_twice = (counts == 2).sum() / total_unique * 100
    string = f"Percentage of people appearing exactly twice: {percent_twice:.2f}%"
    print(string)
    with open(descriptives_file, "w") as f:
        f.write(string + "\n")

    # b) what % reported 3 categories
    percent_thrice = (counts == 3).sum() / total_unique * 100
    string = f"Percentage of people appearing exactly three times: {percent_thrice:.2f}%"
    print(string)
    with open(descriptives_file, "a") as f:
        f.write(string + "\n")


    # for completeness, % who appear exactly once
    percent_once = (counts == 1).sum() / total_unique * 100
    string = f"Percentage of people appearing exactly once: {percent_once:.2f}%"
    print(string)
    with open(descriptives_file, "w") as f:
        f.write(string + "\n")

    # c) total reported acrorss whole sample
    string = f"Total activities across whole sample: {df.shape[0]:.0f}"
    print(string)
    with open(descriptives_file, "a") as f:
        f.write(string + "\n")

    # d) % female : male
    unique_people = df.drop_duplicates(subset='ID_new')
    gender_counts = unique_people['sex'].value_counts(normalize=True) * 100
    string = f"Female: {gender_counts.iloc[0]:.2f}%; Male: {gender_counts.iloc[1]:.2f}%"
    print(string)
    with open(descriptives_file, "a") as f:
        f.write(string + "\n")

    # e) what % had atleast a high school diploma
    edu_counts = unique_people['edu'].value_counts(normalize=True) * 100
    string = f"Education % for category 1: {edu_counts.iloc[1]:.2f}%; category 2 (atleast highschool diploma): {edu_counts.iloc[0]:.2f}%; and missing info: {edu_counts.iloc[2]:.2f}%;"
    print(string)
    with open(descriptives_file, "a") as f:
        f.write(string + "\n")

    # f) what % did less than 75 minutes of exercise/sport a week
    ex_mins_counts = unique_people['sport_min_kat'].value_counts(normalize=True) * 100
    string = (
    "Exercise mins. per week counts (% for unique individuals):\n"
    + ex_mins_counts.round(2).to_string())
    print(string)
    with open(descriptives_file, "a") as f:
        f.write(string + "\n")

    # g) average and SD age
    original_count = len(unique_people)
    removed_df = unique_people[(unique_people['age'] < 18) | (unique_people['age'] > 65)]
    df_filtered = unique_people[(unique_people['age'] >= 18) & (unique_people['age'] <= 65)]
    filtered_count = len(df_filtered)
    removed_count = original_count - filtered_count
    print(f"Removed {removed_count} individuals who's age was > 65 or <18.")
    print(removed_df['age'].value_counts().sort_index())  # To see frequency of each removed age
    mean_age = df_filtered['age'].mean()
    sd_age = df_filtered['age'].std()
    string = f"Mean age: {mean_age:.2f} (SD= {sd_age:.2f})"
    print(string)
    with open(descriptives_file, "a") as f:
        f.write(string + "\n")

    # save preprocessed data:
    df.to_csv("Data/Preprocessed/preprocessed_{}.csv".format(outcome))
    print(f"Preprocessing finished.\nFinal processed data shape: {df.shape}")

    return df



def check_cors(X, save_path, save_name):
    corr_iv_m = X.corr(method='pearson', min_periods=100).abs()
    # drop repetitious pairs (diagonals and below in matrix):
    corr_iv = (corr_iv_m.where(np.triu(np.ones(corr_iv_m.shape), k=1).astype(bool))
               .stack()
               .sort_values(ascending=False))
    # save as a df
    corr_iv = pd.DataFrame(corr_iv)
    # rename col
    corr_iv.columns = ["Absolute_corr"]
    corr_iv.to_csv(save_path + save_name)
    return corr_iv


def remove_cols(df, drop_cols):
    for col in df:
        if col in drop_cols:
            df.drop(col, inplace=True, axis=1)
    return df


def get_preprocessed_col_names(X, pipe, cat_vars):
    numeric_features = X.drop(cat_vars, inplace=False, axis=1).columns
    dum_names = list(pipe.named_steps['preprocessor'].transformers_[1][1].named_steps['oh_encoder']
                     .get_feature_names_out(cat_vars))
    num_names = list(X[numeric_features].columns.values)
    names = num_names + dum_names
    return names


def get_preprocessed_col_names_sm(X, pipe, cat_vars):
    """
    X: original DataFrame before preprocessing
    pipe: a fitted MultiOutputClassifier(...(imbpipeline([...])))
    cat_vars: list of categorical columns used in one-hot encoding
    """

    # 1) After pipe.fit(X, y), each output column has a fitted pipeline in pipe.estimators_
    # We'll just use the first fitted pipeline:
    fitted_pipeline = pipe.estimators_[0]

    # 2) Access the preprocessor inside that pipeline
    preprocessor = fitted_pipeline.named_steps['preprocessor']

    # 3) Identify numeric feature names (assume all non-categorical are numeric)
    numeric_features = [col for col in X.columns if col not in cat_vars]

    # 4) For a ColumnTransformer with two parts: [0] numeric, [1] categorical
    cat_pipeline = preprocessor.transformers_[1][1]  # or find the correct index
    oh_encoder = cat_pipeline.named_steps['oh_encoder']

    # 5) Get the OHE feature names
    ohe_names = oh_encoder.get_feature_names_out(cat_vars)

    # 6) Combine numeric + OHE names
    names = list(numeric_features) + list(ohe_names)

    return names


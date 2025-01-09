import pyreadstat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Missings: an empty variable means that the person did not give an answer to this item.
# A value of -999 means that this item was not asked.
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from fixed_params import outcome, x_lab, goal_vars, person_id, answer_id

from Functions.nmf import Impute, Normalise, NMF_grid, Get_H_matrix, Get_W_matrix, Force_H_matrix
from Functions.plotting import plot_count, plot_perc, plot_by_var, plot_clust, plot_clust_col, find_K, Plot_NMF_all

df, meta = pyreadstat.read_sav('Data/FinalDaten_BMZI_W1_W2_long.sav')

# predicting cat sport b from goal variables
other_vars = ['sex', 'age', 'edu', 'sport_min_kat']
X = df[[person_id] + [answer_id] + other_vars + goal_vars]

# code "-999" as na
df[outcome] = np.where(df[outcome] == -999.0, np.nan, df[outcome])
y = df[outcome].astype('category')

# descriptives of outcome
print(y.value_counts())
order_num = list(y.value_counts().index)
order = y.nunique()

# create data
X_and_y = pd.concat([X, y], axis=1)
X_and_y.sort_values(by=["ID_new", "Index1"], inplace=True, axis=0)
X_and_y.to_csv("Data/X_and_y_{}.csv".format(outcome))

cat_names = list(meta.variable_value_labels[outcome].values())
cat_nums = list(meta.variable_value_labels[outcome].keys())
short_names = []
very_short_names = []
for name in cat_names:
    short_name = name.split("/", 1)[0]
    very_short_name = name.split(" ", 1)[0]
    short_names.append(short_name)
    very_short_names.append(very_short_name)

# Create a dictionary by zipping the two lists together
numbers = range(1, 11)
very_short_names_dict = dict(zip(numbers, very_short_names))
short_names_dict = dict(zip(numbers, short_names))

save_meta = "Data/Meta/"
pd.DataFrame(cat_names).to_csv(save_meta + "full_outcome_names_{}.csv".format(outcome))
pd.DataFrame(short_names).to_csv(save_meta + "short_outcome_names_{}.csv".format(outcome))
pd.DataFrame(very_short_names).to_csv(save_meta + "v_short_outcome_names_{}.csv".format(outcome))

# plots

# plot freq of categories
plot_count(data=X_and_y, x=outcome, hue=outcome, xlabs=very_short_names,
           save_path="Outputs/Descriptives/", save_name = "y_hist_{}".format(outcome),
           xlab=x_lab, title="Distribution of outcome variable",
           order=cat_nums, leg_title=x_lab, label_dict=short_names_dict)

# plot freq of categories by gender
plot_by_var(data=X_and_y, x=outcome, hue="sex", xlabs=very_short_names,
           save_path="Outputs/Descriptives/", save_name = "y_hist_gender_{}".format(outcome),
           xlab=x_lab, leg_labs=["Women", "Men"],
            title="Distribution of outcome variable by gender")

# plot % of categories
plot_perc(data=X_and_y, x=outcome, hue=outcome, xlabs=very_short_names,
           save_path="Outputs/Descriptives/", save_name = "y_perc_{}".format(outcome),
           xlab=x_lab, title="Distribution of outcome variable (%)",
           order=cat_nums, leg_title=x_lab, label_dict=short_names_dict)

# todo: reverse colours on legend
# ------------------------------------------------------------------------------------------

# Cluster
df.dropna(axis=0, how="all", subset=goal_vars, inplace=True)
df.dropna(subset=[outcome], axis=0, inplace=True, how='any')
X = df[other_vars + goal_vars]
X = Impute(X)
X = Normalise(X)

y = df[outcome].astype('category')

save_path="Outputs/Descriptives/Clusters/"
# find_K(X, save_path, "find_K")

n_clust = 6
km = KMeans(n_clusters=n_clust, random_state=93).fit(X)
labs = km.labels_
kmeans = KMeans(n_clusters=n_clust, random_state=93).fit_transform(X)
plot_clust(kmeans=kmeans, y=y, save_path=save_path, save_name="clusters_{}".format(n_clust))
plot_clust_col(kmeans=kmeans, labs=labs, n_clust=n_clust, save_path=save_path,
               save_name="clusters_col_{}".format(n_clust))

# NMF
save_path="Outputs/Descriptives/NMF/"
param_save_name = 'Outputs/Descriptives/NMF/NMF_params.pickle'
# NMF_grid(X, y, param_save_name)

nmf_opt = NMF(init="random", solver="cd", max_iter=250, random_state=93, alpha_W=5, n_components=5, tol=0.007)
nmf_opt.fit(X, y)
print(nmf_opt.reconstruction_err_)

H = Get_H_matrix(X, param_save_name, save_path, "H")
W = Get_W_matrix(X, param_save_name, save_path, "W")

H_f = Force_H_matrix(X, save_path, save_name="H_f", best_n=5, best_tol=0.001, best_alpha=2)

Plot_NMF_all(H, save_path, save_name="H_all")
Plot_NMF_all(H_f, save_path, save_name="H_f")

print('done')

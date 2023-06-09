import pyreadstat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Missings: an empty variable means that the person did not give an answer to this item.
# A value of -999 means that this item was not asked.
from Functions.plotting import plot_count, plot_perc, plot_by_var

df, meta = pyreadstat.read_sav('Data/all_long.sav')

# predicting cat sport b from goal variables
outcome = "sport_kat_b"
person_id = df['ID_new']
goal_vars = list(df.iloc[:, 53:61].columns)
other_vars = ['sex', 'age', 'edu', 'sport_minwk', 'sport_min_kat']
X = df[other_vars + goal_vars]
y = df[outcome].astype('category')

# descriptives on outcome
print(y.value_counts())
order_num = list(y.value_counts().index)
order = y.nunique()

# create data
X_and_y = pd.concat([X, y], axis=1)
X_and_y.to_csv("Data/X_and_y.csv")

cat_names = list(meta.variable_value_labels["sport_kat_b"].values())
cat_nums = list(meta.variable_value_labels["sport_kat_b"].keys())
short_names = []
very_short_names = []
for name in cat_names:
    short_name = name.split("/", 1)[0]
    very_short_name = name.split(" ", 1)[0]
    short_names.append(short_name)
    very_short_names.append(very_short_name)

save_meta = "Data/Meta/"
pd.DataFrame(cat_names).to_csv(save_meta + "full_outcome_names.csv")
pd.DataFrame(short_names).to_csv(save_meta + "short_outcome_names.csv")
pd.DataFrame(very_short_names).to_csv(save_meta + "v_short_outcome_names.csv")

# plot freq of categories
plot_count(data=X_and_y, x=outcome, hue=outcome, xlabs=very_short_names,
           save_path="Outputs/Descriptives/", save_name = "y_hist",
           xlab="Sport Category (B)", leg_labs=short_names, title="Distribution of outcome variable")

# plot freq of categories by gender
plot_by_var(data=X_and_y, x=outcome, hue="sex", xlabs=very_short_names,
           save_path="Outputs/Descriptives/", save_name = "y_hist_gender",
           xlab="Sport Category (B)", leg_labs=["Women", "Men"],
            title="Distribution of outcome variable by gender")

# plot % of categories
plot_perc(data=X_and_y, x=outcome, hue=outcome, xlabs=very_short_names,
           save_path="Outputs/Descriptives/", save_name = "y_perc",
           xlab="Sport Category (B)", leg_labs=short_names, title="Distribution of outcome variable (%)",
          order=cat_nums)
# todo: reverse colours on legend

print('done')


import json
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from Functions.read import load_f1_scores
from fixed_params import outcome
from Functions.plotting import plot_f1_scores


start_string = "_24_Jan_2025__17.50"
model = "GB"
load_base_results_path = "Results/Prediction/Baseline_per_category/"
load_pred_results_path = "Results/Prediction/"
model_results_file = f"Results/Prediction/{model}_{start_string}.txt"

with open(f"{load_base_results_path}all_base_scores_{outcome}{start_string}.json", "r") as json_file:
    base_scores = json.load(json_file)

with open("Data/Dicts_and_Lists/short_names_dict.json", "r") as json_file:
    short_names_dict = json.load(json_file)

results = pd.read_csv(f"{load_pred_results_path}all_test_scores_{outcome}{start_string}.csv")

# Load the model's f1-scores
gb_f1_scores = load_f1_scores(model_results_file)

# Step 3: Organize Data into a DataFrame
categories = [str(i) for i in range(10)]
df = pd.DataFrame(index=categories)

# Add baseline models' f1-scores
for model in ['Dummy_MF', 'Dummy_Stratified', 'Dummy_Random']:
    df[model] = df.index.map(lambda x: base_scores[model].get(x, {}).get('f1-score', 0.0))

# Add GB model's f1-scores
df['GB'] = df.index.map(lambda x: gb_f1_scores.get(x, 0.0))

# Optional: Sort the DataFrame by category
df = df.sort_index()

# Plot the f1-scores
plot_f1_scores(df,
               save_path="Results/Prediction/Plots/Per_category/",
               save_name=f"f1_score_per_category_{outcome}{start_string}.png",
               cat_labels=short_names_dict)

print('done')
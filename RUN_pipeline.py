# from Functions.plotting import plot_results
from Functions.plotting import plot_results
from Functions.prediction import prediction
import pandas as pd

from Functions.preprocess import preprocess
from fixed_params import outcome

df = pd.read_csv("Data/X_and_y.csv", index_col=[0])

# (1) preprocess

df = preprocess(df, outcome)

# (2) prediction
optimised_pipes = prediction(outcome=outcome, df=df, test_run=True,
                             use_pre_trained=False, smote=True)

# (3) plot prediction results
results_df = pd.read_csv("Results/Prediction/all_test_scores.csv")
results_df.rename(columns={"Unnamed: 0": "Model"}, inplace=True)
save_path = "Results/Prediction/Plots/"
save_name = "all_prediction_results"
x_ticks = ["Dummy Most Frequent", "Dummy Random",
           "Dummy Stratified", "Logistic Regression",
           "Elastic Net", "Random Forest"]

f1_weight = results_df[results_df.Model == "F1_weighted"]
log_loss = results_df[results_df.Model == "Log_loss"]

save_name = "all_predict_f1weight"
plot_results(y="F1_weighted", data=f1_weight, colour='Model',
             save_path=save_path, save_name=save_name,
             xlab="Prediction Models", ylab="F1 weighted",
             title="Comparison of Predictions",
             x_ticks=x_ticks
             )

save_name = "all_predict_logloss"
plot_results(y="Log_loss", data=log_loss, colour='Model',
             save_path=save_path, save_name=save_name,
             xlab="Prediction Models", ylab="Log loss",
             title="Comparison of Predictions",
             x_ticks=x_ticks
             )

# # (4) interpretation
# interpretation(dv=dv, df=df, analysis=analysis, m=m,
#                optimised_pipes=optimised_pipes, test_run=test_run,
#                start_string=start_string)

print('done')
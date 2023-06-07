# from Functions.plotting import plot_results
from Functions.prediction import prediction
import pandas as pd

from Functions.preprocess import preprocess
from fixed_params import outcome

df = pd.read_csv("Data/X_and_y.csv", index_col=[0])

# (1) preprocess

df = preprocess(df, outcome)

# (2) prediction
optimised_pipes = prediction(outcome=outcome, df=df, test_run=True,
                             use_pre_trained=False)
#
# # (3) plot prediction results
# results_df = pd.read_csv("{}/Results{}/Prediction/all_test_scores_{}{}.csv".format(analysis, m, start_string, t))
# save_path = "{}/Results{}/Prediction/Plots/".format(analysis, m)
# save_name = "all_prediction_results_{}{}".format(start_string, t)
# x_ticks = ["Elastic Net", "Random Forest", "Gradient Boosting"]
# plot_results(y="R2", data=results_df, colour='Model',
#              save_path=save_path, save_name=save_name,
#              xlab="Prediction Models", ylab="Prediction R Squared",
#              title="Comparison of Predictions",
#              x_ticks=x_ticks
#              )

# # (4) interpretation
# interpretation(dv=dv, df=df, analysis=analysis, m=m,
#                optimised_pipes=optimised_pipes, test_run=test_run,
#                start_string=start_string)

print('done')

import pandas as pd
import datetime as dt
from Functions.plotting import run_plots, run_plots_multilabel
from Functions.prediction import prediction
from Functions.preprocess import preprocess
from fixed_params import outcome, multi_label

use_pre_trained = False
test_run = False

df = pd.read_csv("Data/X_and_y_{}.csv".format(outcome), index_col=[0])

# (1) preprocess

df = preprocess(df, outcome)

# (2) prediction
start = dt.datetime.now()
if use_pre_trained == False:
    start_string = start.strftime('_%d_%b_%Y__%H.%M')
else:
    start_string = "_11_Jun_2023__07.44"
if test_run == True:
    t= "_test"
else:
    t= ""

optimised_pipes = prediction(outcome=outcome, df=df, test_run=test_run,
                             use_pre_trained=use_pre_trained, smote=False,
                             start_string=start_string, t=t, multi_label=multi_label)

# (3) plot prediction results
results_df = pd.read_csv("Results/Prediction/all_test_scores_{}{}{}.csv".format(outcome, start_string, t))
if multi_label == False:
    run_plots(results_df, start_string, t)
else:
    run_plots_multilabel(results_df, start_string, t)


# # (4) interpretation
# interpretation(dv=dv, df=df, analysis=analysis, m=m,
#                optimised_pipes=optimised_pipes, test_run=test_run,
#                start_string=start_string)

print('done')

import pandas as pd
import datetime as dt

from Functions.interpretation import interpretation
from Functions.plotting import run_plots, run_plots_multilabel
from Functions.prediction import prediction
from Functions.preprocessing import preprocess
from fixed_params import outcome, multi_label, smote

use_pre_trained = True
test_run = False
do_testset_evaluation = False


df = pd.read_csv("Data/X_and_y_{}.csv".format(outcome), index_col=[0])

# (1) preprocess

df = preprocess(df, outcome)

# (2) prediction
start = dt.datetime.now()
if use_pre_trained == False:
    start_string = start.strftime('_%d_%b_%Y__%H.%M')
else:
    # start_string = "_11_Jun_2023__07.44"
    start_string = "_19_Aug_2023__20.50"
if test_run == True:
    t= "_test"
else:
    t= ""

optimised_pipes = prediction(outcome=outcome, df=df, test_run=test_run,
                             use_pre_trained=use_pre_trained, smote=smote,
                             start_string=start_string, t=t, multi_label=multi_label,
                             do_testset_evaluation=do_testset_evaluation,
                             predict_probab=False)

# (3) plot prediction results
results_df = pd.read_csv("Results/Prediction/all_test_scores_{}{}{}.csv".format(outcome, start_string, t))
if multi_label == False:
    run_plots(results_df, start_string, t)
else:
    run_plots_multilabel(results_df, start_string, t)

# todo: add option to turn off pred prob

# (4) interpretation
interpretation(outcome=outcome, df=df,
               optimised_pipes=optimised_pipes,
               start_string=start_string, t=t,
               do_impurity_importance=True,
               do_permutation_importance=False,
               do_SHAP_importance=True,
               recalc_SHAP=True
               )

print('done')
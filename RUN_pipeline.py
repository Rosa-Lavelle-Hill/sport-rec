
import pandas as pd
import datetime as dt

from Functions.interpretation import interpretation
from Functions.plotting import run_plots, run_plots_multilabel
from Functions.prediction import prediction
from Functions.preprocessing import preprocess
from fixed_params import outcome, multi_label, smote, single_label_scoring_name, multi_label_scoring_name

# Run options:
use_pre_trained = True
test_run = False
do_testset_evaluation = True
only_best_model = True
# ==========================
do_Enet = True
do_GB = True

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("Data/X_and_y_{}.csv".format(outcome), index_col=[0])

    # (1) preprocess

    df = preprocess(df, outcome)

    # (2) prediction
    start = dt.datetime.now()
    if use_pre_trained == False:
        start_string = start.strftime('_%d_%b_%Y__%H.%M')
    else:
        # start_string = "_11_Jun_2023__07.44"
        # start_string = "_19_Aug_2023__20.50"
        # start_string = "_10_Jan_2025__17.00"
        start_string = "_24_Jan_2025__17.50"

    if test_run == True:
        t= "_test"
    else:
        t= ""

    optimised_pipes, model_names = prediction(outcome=outcome,
                                 df=df,
                                 test_run=test_run,
                                 use_pre_trained=use_pre_trained,
                                 smote=smote,
                                 start_string=start_string,
                                 t=t,
                                 multi_label=multi_label,
                                 do_testset_evaluation=do_testset_evaluation,
                                 predict_probab=True,
                                 do_Enet=do_Enet,
                                 do_GB=do_GB
                                 )

    # (3) plot prediction results
    results_df = pd.read_csv("Results/Prediction/all_test_scores_{}{}{}.csv".format(outcome, start_string, t))

    if multi_label == False:
        run_plots(results_df=results_df,
                             start_string=start_string,
                             t=t,
                             do_Enet=do_Enet,
                             do_GB=do_GB)
    else:
        run_plots_multilabel(results_df=results_df,
                             start_string=start_string,
                             t=t,
                             do_Enet=do_Enet,
                             do_GB=do_GB)

    # todo: plot by category, best model compared to baselines

    # (4) interpretation
    if only_best_model == True:
        if multi_label == True:
            select_model_score = multi_label_scoring_name
        if multi_label == False:
            select_model_score = single_label_scoring_name
        # Select the column with the highest value for select_model_score
        row = results_df[results_df["Model"] == select_model_score]
        row_models = row[model_names]
        max_values = row_models.max()
        best_model = max_values[max_values == max_values.max()].index[-1]
        print(f"Only interpreting column with the highest {select_model_score}: {best_model}")
    else:
        best_model = None

    interpretation(outcome=outcome,
                   df=df,
                   optimised_pipes=optimised_pipes,
                   start_string=start_string,
                   t=t,
                   do_impurity_importance=False,
                   do_permutation_importance=True,
                   do_SHAP_importance=True,
                   recalc_SHAP=False,
                   model_names=model_names,
                   best_model=best_model,
                   load_fitted_model=True,
                   do_shap_force_plot=False,
                   interaction_plots=True,
                   load_explainer=False
                   )

    end = dt.datetime.now()
    runtime = end - start
    print(f'Done. Run time: {runtime}')
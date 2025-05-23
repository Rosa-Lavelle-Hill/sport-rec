from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

outcome = "sport_kat_d2"
if outcome == "sport_kat_b":
    x_lab = "Sport Category (B)"
if outcome == "sport_kat_c":
    x_lab = "Sport Category (C)"
if outcome == "sport_kat_d2":
    x_lab = "Sport Category (D2)"
person_id = 'ID_new'
answer_id = "Index1"

single_label_scoring = "f1_micro"
single_label_scoring_name = "micro_f1"
multi_label_scoring = make_scorer(f1_score, average="micro", zero_division=0)
multi_label_scoring_name = "micro_f1"

multi_label = True
smote = True
# predict_probab = True

decimal_places = 3
verbose = 2
random_state = 93
test_size = 0.2
nfolds = 3 # so enough positive examples of smaller categories in each fold
imputer_max_iter = 10
n_shap_features = 15
n_permutations = 3

categorical_features = ['edu', 'sex', 'sport_min_kat']
goal_vars = ['Zind_fitheal', 'Zind_figap', 'Zind_disstre', 'Zind_actenj', 'Zind_compperf', 'Zind_aes', 'Zind_con']

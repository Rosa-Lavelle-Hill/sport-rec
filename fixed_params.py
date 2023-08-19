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
multi_label_scoring = make_scorer(f1_score, average="micro", zero_division=0)

multi_label = True
smote = False
do_Enet = False
do_GB = False

decimal_places = 2
verbose = 2
random_state = 93
nfolds = 5
imputer_max_iter = 10

categorical_features = ['edu', 'sex', 'sport_min_kat']
goal_vars = ['Zind_fitheal', 'Zind_figap', 'Zind_disstre', 'Zind_actenj', 'Zind_compperf', 'Zind_aes', 'Zind_con']

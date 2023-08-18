outcome = "sport_kat_d2"
if outcome == "sport_kat_b":
    x_lab = "Sport Category (B)"
if outcome == "sport_kat_c":
    x_lab = "Sport Category (C)"
if outcome == "sport_kat_d2":
    x_lab = "Sport Category (D2)"
person_id = 'ID_new'
answer_id = "Index1"

decimal_places = 2
single_label_scoring = "f1_weighted"
multi_label = True
smote = True
verbose = 2
random_state = 93
nfolds = 3
categorical_features = ['edu', 'sex', 'sport_min_kat']
imputer_max_iter = 5
goal_vars = ['Zind_fitheal', 'Zind_figap', 'Zind_disstre', 'Zind_actenj', 'Zind_compperf', 'Zind_aes', 'Zind_con']

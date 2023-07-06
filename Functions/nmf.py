import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from imblearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.decomposition import NMF
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from matplotlib import cm, colors


def Plot_Basis(H, X):
    color = cm.inferno_r(np.linspace(.2, .8, 20))
    H_trans = pd.DataFrame(Normalise(np.transpose(H)))

    l = list()
    for i in list(range(0, len(H_trans.columns))):
        l.append('V' + str(i))
    H_trans.columns = l

    d = {}
    colnames = list(X.columns.values)
    for col in range(0, len(H_trans.columns)):
        d[col] = pd.DataFrame(H_trans.iloc[:, col])
        d[col]['Var'] = colnames
        d[col] = d[col].sort_values(d[col].columns.values[0], axis=0, ascending=True)
        d[col] = d[col].iloc[len(H_trans) - 10:len(H_trans), :]

        plt.barh(d[col]['Var'], d[col].iloc[:, 0], color=color)
        plt.xticks(rotation=90)
        plt.suptitle('Basis ' + str(d[col].columns.values[0])[1:])
        plt.show()


def Get_W_matrix(X, file_name, save_path, save_name):
    with open(file_name, 'rb') as handle:
        b = pickle.load(handle)
    best_tol = b["nmf__tol"]
    best_alpha = b["nmf__alpha"]
    best_n = b["nmf__n_components"]
    nmf_opt = NMF(init="random", solver="cd", max_iter=200, n_components=best_n, tol=best_tol, alpha=best_alpha,
                  random_state=93)
    W = pd.DataFrame(nmf_opt.fit_transform(X))
    W.to_csv(save_path + save_name + '.csv')
    return W


def Get_H_matrix(X, file_name, save_path, save_name):
    with open(file_name, 'rb') as handle:
        b = pickle.load(handle)
    best_tol = b["nmf__tol"]
    best_alpha = b["nmf__alpha"]
    best_n = b["nmf__n_components"]
    nmf_opt = NMF(init="random", solver="cd", max_iter=200, n_components=best_n, tol=best_tol, alpha=best_alpha,
                  random_state=93)
    nmf_opt.fit_transform(X)
    H = pd.DataFrame(nmf_opt.components_)
    H.columns = list(X.columns.values)
    H.to_csv(save_path + save_name + '.csv')
    return H


def Force_H_matrix(X, save_path, save_name, best_n, best_tol, best_alpha):
    nmf_opt = NMF(init="random", solver="cd", max_iter=200, n_components=best_n, tol=best_tol, alpha=best_alpha,
                  random_state=93)
    nmf_opt.fit_transform(X)
    H = pd.DataFrame(nmf_opt.components_)
    H.columns = list(X.columns.values)
    H.to_csv(save_path + save_name + '.csv')
    return H


def NMF_grid(X, y, save_name):
    n_comp = [5, 6, 7, 8]
    tol = np.linspace(0.001, 0.01, 4)
    alpha = [1, 2, 5]
    seed = 93
    cv = 3
    clf = Pipeline([
        ("nmf", NMF(init="random", solver="cd", max_iter=200, random_state=seed)),
        ("lin_reg", LogisticRegression())
    ])
    param_grid = [{"nmf__n_components": n_comp,
                   "nmf__tol": tol,
                   "nmf__alpha": alpha
                   }]
    grid_search = GridSearchCV(clf, param_grid, cv=cv)
    grid_search.fit(X, y)
    with open(save_name, 'wb') as handle:
        pickle.dump(grid_search.best_params_, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("NMF best parameters are:" + str(grid_search.best_params_))
    return


def Normalise(X):
    range = (0, 1)
    a = X.values
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=range)
    a = min_max_scaler.fit_transform(a)
    X = pd.DataFrame(a, columns=X.columns)
    return X


def Impute(X):
    Seed = 93
    max_iter = 10
    imp = IterativeImputer(max_iter=max_iter, random_state=Seed)
    imp.fit(X)
    X = pd.DataFrame(imp.transform(X), columns=X.columns)
    return X
import seaborn as sns
import matplotlib.pyplot as plt
import colorcet as cc
import shap
import pandas as pd
import numpy as np
from matplotlib import colors
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from fixed_params import do_Enet, do_GB

def plot_count_unordered(data, x, hue, xlabs, save_path, save_name, xlab, leg_labs, leg_title,
               title=""):
    n = data[x].nunique() + 1
    plt.figure(figsize=(7.8, 4))
    ax = sns.histplot(data=data, x=x, hue=hue, stat="count", discrete=True, bins=14,
                 legend=False)
    plt.xticks(ticks=range(1, n), labels=xlabs, rotation=60, size=7)
    plt.xlabel(xlab)
    plt.title(title)
    plt.subplots_adjust(right=0.3)
    plt.legend(bbox_to_anchor=(1, 1), title=xlab, loc='upper left',
               labels=leg_labs, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return



def plot_count(data, x, hue, xlabs, save_path, save_name, xlab,
              label_dict, order, leg_title, title=""):
    n = data[x].nunique() + 1
    plt.figure(figsize=(7.8, 4))
    ax = sns.histplot(data=data, x=x, hue=hue, stat="count", discrete=True, bins=14, hue_order=order)
    plt.xticks(ticks=range(1, n), labels=xlabs, rotation=60, size=7)
    plt.xlabel(xlab)
    plt.title(title)

    # Create a manual legend
    unique_hues = data[hue].unique()
    handles = [plt.Line2D([0], [0], marker='o', color=sns.color_palette()[i], linestyle='')
               for i, hue_val in enumerate(order) if hue_val in unique_hues]
    new_labels = [label_dict.get(hue_val, hue_val) for hue_val in order if hue_val in unique_hues]

    plt.legend(handles, new_labels, title=leg_title, loc='upper right', fontsize=9)

    plt.subplots_adjust(right=0.3)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()


def plot_perc(data, x, hue, xlabs, save_path, save_name, xlab,
              label_dict, order, leg_title, title=""):
    n = data[x].nunique() + 1
    plt.figure(figsize=(7.8, 4))
    ax = sns.histplot(data=data, x=x, hue=hue, stat="percent", discrete=True, bins=14, hue_order=order)
    plt.xticks(ticks=range(1, n), labels=xlabs, rotation=60, size=7)
    plt.xlabel(xlab)
    plt.title(title)

    # Create a manual legend
    unique_hues = data[hue].unique()
    handles = [plt.Line2D([0], [0], marker='o', color=sns.color_palette()[i], linestyle='')
               for i, hue_val in enumerate(order) if hue_val in unique_hues]
    new_labels = [label_dict.get(hue_val, hue_val) for hue_val in order if hue_val in unique_hues]

    plt.legend(handles, new_labels, title=leg_title, loc='upper right', fontsize=9)

    plt.subplots_adjust(right=0.3)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()


def plot_by_var(data, x, hue, xlabs, save_path, save_name, xlab, leg_labs, leg_title='Gender', title=""):
    n = data[x].nunique() + 1
    plt.figure(figsize=(7.8, 4))
    sns.histplot(data=data, x=x, hue=hue, stat="count", discrete=True, bins=14,
                 legend=False)
    plt.xticks(ticks= range(1, n), labels=xlabs, rotation=60, size=7)
    plt.legend(bbox_to_anchor=(1, 1), title=leg_title, loc='upper left',
               labels=leg_labs, fontsize=9)
    plt.xlabel(xlab)
    plt.title(title)
    plt.subplots_adjust(right=0.3)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return



def plot_precis_recall_curve(yhat, y_test, y, model_lab):
    """Plots a precision-recall curve."""
    pos_probs = yhat[:, 1]
    no_skill = len(y[y == 1]) / len(y)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    precision, recall, _ = precision_recall_curve(y_test, pos_probs)
    plt.plot(recall, precision, marker='.', label=model_lab)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()



def plot_roc(probs, y_test, model_name):
    """Plots a ROC curve and compares to a 'no skill' line."""
    # keep probabilities for the positive outcome only
    lr_probs = probs[:, 1]
    ns_probs = [0 for _ in range(len(y_test))]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('{}: ROC AUC=%.3f'.format(model_name) % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    plt.figure(figsize=(6, 6))
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label=model_name)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()


def plot_results(y, data, colour, save_path, save_name,
                               xlab, ylab, title, x_ticks, legend=False,
                               fontsize=12, legend_pos="lower right"):
    palette = ["plum", "cornflowerblue", "coral", "mediumaquamarine", "peru", "khaki"]

    sns.set_palette(palette)
    fig, ax = plt.subplots()
    plt.figure(figsize=(6, 4))

    data = data.transpose().reset_index()
    data.columns=data.iloc[0]
    data.drop(index=data.index[0], axis=0, inplace=True)
    data.rename(columns={'Unnamed: 0': 'Model'}, inplace=True)

    g=sns.barplot(x="Model", y=y, data=data, hue=colour, dodge=False,
                  palette=sns.color_palette(palette, 6))

    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)

    g.set_xticklabels(x_ticks)
    plt.xticks(rotation=70)

    plt.tight_layout()
    if legend == True:
        plt.legend(loc=legend_pos)
    else:
        plt.legend([], [], frameon=False)
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()


def plot_impurity(impurity_imp_df, save_path, save_name):
    y_ticks = np.arange(0, impurity_imp_df.shape[0])
    fig, ax = plt.subplots()
    ax.barh(y_ticks, impurity_imp_df["Importance"])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(impurity_imp_df["Feature"])
    ax.set_title("Impurity Reduction Importance (training set)")
    fig.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    return


def plot_impurity_ml(impurity_imp_df, save_path, save_name):
    # takes the mean
    plt.figure(figsize=(8, 8))
    impurity_imp_df.mean().sort_values(ascending=False).plot(kind='bar')
    plt.title('Average feature importance across all sport classifications')
    plt.xlabel('Features')
    plt.ylabel('Impurity Importance')
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    return


def heatmap_importance(df, save_name, save_path,  title,
                       xlab, ylab, fontsize=12, palette="viridis", show_n="all",
                       figsize=(8, 9), sort_by = "overall", tick_font_size=9):
    df = np.transpose(df)
    if sort_by == "overall":
        df['sum'] = df.sum(axis=1)
        df.sort_values(by="sum", inplace=True, ascending=False)
        df.drop("sum", axis=1, inplace=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, cmap=palette, yticklabels=True,
                cbar_kws={'shrink': 0.5})
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.yticks(fontsize=tick_font_size)
    # plt.subplots_adjust(left=0.1)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    return



def plot_permutation(perm_imp_df, save_path, save_name):
    y_ticks = np.arange(0, perm_imp_df.shape[0])
    fig, ax = plt.subplots()
    ax.barh(y_ticks, perm_imp_df["Importance"])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(perm_imp_df["Feature"])
    ax.set_title("Permutation Importance (test set)")
    fig.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    return


def plot_label_scat(x, y, country, x_lab, y_lab,
                    save_path, save_name, c):
    plt.figure(figsize=(20, 20))
    n = list(country)
    fig, ax = plt.subplots()
    plt.scatter(x, y,
                c=c, edgecolor='none', alpha=1.5, s=10,
                cmap=plt.cm.get_cmap("spring", 4))
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.colorbar()

    # for i, txt in enumerate(n):
    #     plt.annotate(txt, (np.array(x)[i], np.array(y)[i]))
    fig.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    return



def plot_label_reg_sns(x, y, country, x_lab, y_lab,
                    save_path, save_name, anov_var='Country_Name',
                       cor=True, ano=True):
    plt.figure(figsize=(20, 20))
    n = list(anov_var)
    fig, ax = plt.subplots()
    sns.regplot(x=x, y=y)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    if ano == True:
        for i, txt in enumerate(n):
            plt.annotate(txt, (np.array(x)[i], np.array(y)[i]))
    if cor == True:
        # computes correlations excluding NAs:
        x_str = x.name
        y_str = y.name
        corr = round(pd.Series(x).corr(pd.Series(y), method='pearson', min_periods=None), 2)
        print('correlation between x ({}) and y ({}) = {}'.format(x_str, y_str, corr))
        if len(list(country.unique())) == 1:
            title = "{}: Pearson r Correlation = {}".format(np.array(country)[0], corr)
        else:
            title = "Pearson r Correlation = {}".format(corr)
        plt.title(title, fontsize=12)
    fig.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    return


def plot_SHAP(shap_dict, col_list, plot_type, n_features,
              save_path, save_name, figsize=(6,6)):
    plt.figure(figsize=figsize)
    shap.summary_plot(shap_dict, feature_names=col_list, show=False,
                      plot_type=plot_type, max_display=n_features)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()


def plot_SHAP_df(shap_array, X_df, col_list, plot_type, n_features, title,
              save_path, save_name, figsize=(6,6)):
    plt.figure(figsize=figsize)
    shap.summary_plot(shap_array, X_df, feature_names=col_list, show=False,
                      plot_type=plot_type, max_display=n_features, title=title)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()



def plot_forceSHAP_df(shap_vals_df, explainer, i, base_val, col_list, title,
              save_path, save_name, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    shap_vals = shap_vals_df.iloc[i]
    feature_names = shap_vals_df.columns
    explanation = shap.Explanation(values=shap_vals, base_values=base_val, data=shap_vals_df)
    shap.initjs()
    shap.plots.force(explanation.base_values, explanation.values, feature_names=feature_names, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()



def plot_hist(x, bins, title,
              save_name, save_path,
              xlim=None, ylim=None, color ="skyblue",
              fig_size=(5,5), xlab="", ylab='', fontsize=12):
    """Plots a histogram with a red mean line (+/- 1SD in green dashes)
    and a 95 and 5 percentile dotted lines in blue."""
    plt.figure(figsize=fig_size)
    if xlim != None:
        plt.xlim(xlim)
    if ylim != None:
        plt.ylim(ylim)
    if (x.isnull().all() == False):
        plt.hist(x, bins = bins, color = color, alpha=0.5)
        plt.title(title, fontsize=fontsize)
        plt.xlabel(xlab, fontsize=fontsize)
        plt.ylabel(ylab, fontsize=fontsize)
        if (x.dtype == "float64") or (x.dtype == "float32") or (x.dtype == "int"):
                plt.axvline(x=np.mean(x) - np.std(x), ls="--", color='#2ca02c', alpha=0.7)
                plt.axvline(x=np.mean(x) + np.std(x), ls="--", color='#2ca02c', alpha=0.7)
                plt.axvline(x=np.mean(x), ls="-", color='red', alpha=0.7)
                plt.axvline(x=np.percentile(x, 5), ls="dotted", color='lightblue', alpha=0.7)
                plt.axvline(x=np.percentile(x, 95), ls="dotted", color='lightblue', alpha=0.7)
                plt.tight_layout()
                plt.savefig(save_path+save_name)
        else:
            print("not int or float")
    else:
            print("nas")

    return


def plot_confusion_matrix(cm,
                          target_names,
                          save_path,
                          save_name,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportion
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return


def run_plots(results_df, start_string, t):
    results_df.rename(columns={"Unnamed: 0": "Model"}, inplace=True)
    save_path = "Results/Prediction/Plots/"
    save_name = "all_prediction_results"
    x_ticks = ["Dummy Most Frequent", "Dummy Constant: 0",
               "Dummy Random",
               "Dummy Stratified", "Logistic Regression",
               "Elastic Net", "Random Forest", "Gradient Boosting"]
    if do_Enet == False:
        x_ticks.remove("Elastic Net")
    if do_GB == False:
        x_ticks.remove("Gradient Boosting")

    f1_weight = results_df[results_df.Model == "F1_weighted"]
    log_loss = results_df[results_df.Model == "Log_loss"]

    save_name = "all_predict_f1weight_{}{}".format(start_string, t)
    plot_results(y="F1_weighted", data=f1_weight, colour='Model',
                 save_path=save_path, save_name=save_name,
                 xlab="Prediction Models", ylab="F1 weighted",
                 title="Comparison of Predictions",
                 x_ticks=x_ticks
                 )

    save_name = "all_predict_logloss_{}{}".format(start_string, t)
    plot_results(y="Log_loss", data=log_loss, colour='Model',
                 save_path=save_path, save_name=save_name,
                 xlab="Prediction Models", ylab="Log loss",
                 title="Comparison of Predictions",
                 x_ticks=x_ticks
                 )
    return

def run_plots_multilabel(results_df, start_string, t):
    results_df.rename(columns={"Unnamed: 0": "Model"}, inplace=True)
    save_path = "Results/Prediction/Plots/"
    save_name = "all_prediction_results"
    x_ticks = ["Dummy Most Frequent", "Dummy Constant: 0",
               "Dummy Random",
               "Dummy Stratified", "Logistic Regression",
               "Elastic Net", "Random Forest", "Gradient Boosting"]
    if do_Enet == False:
        x_ticks.remove("Elastic Net")
    if do_GB == False:
        x_ticks.remove("Gradient Boosting")

    micro_precision = results_df[results_df.Model == "micro_precision"]
    micro_f1 = results_df[results_df.Model == "micro_f1"]
    weighted_precision = results_df[results_df.Model == "weighted_precision"]
    weighted_f1 = results_df[results_df.Model == "weighted_f1"]

    save_name = "all_predict_micro_precision_{}{}".format(start_string, t)
    plot_results(y="micro_precision", data=micro_precision, colour='Model',
                 save_path=save_path, save_name=save_name,
                 xlab="Prediction Models", ylab="Micro Precision",
                 title="Comparison of Predictions",
                 x_ticks=x_ticks
                 )

    save_name = "all_predict_micro_f1_{}{}".format(start_string, t)
    plot_results(y="micro_f1", data=micro_f1, colour='Model',
                 save_path=save_path, save_name=save_name,
                 xlab="Prediction Models", ylab="Micro F1",
                 title="Comparison of Predictions",
                 x_ticks=x_ticks
                 )

    save_name = "all_predict_weighted_precision_{}{}".format(start_string, t)
    plot_results(y="weighted_precision", data=weighted_precision, colour='Model',
                 save_path=save_path, save_name=save_name,
                 xlab="Prediction Models", ylab="Weighted Precision",
                 title="Comparison of Predictions",
                 x_ticks=x_ticks
                 )

    save_name = "all_predict_weighted_f1_{}{}".format(start_string, t)
    plot_results(y="weighted_f1", data=weighted_f1, colour='Model',
                 save_path=save_path, save_name=save_name,
                 xlab="Prediction Models", ylab="Weighted F1",
                 title="Comparison of Predictions",
                 x_ticks=x_ticks
                 )
    return


def plot_clust(kmeans, y, save_path, save_name):
    plt.figure(figsize=(7, 5))
    plt.scatter(kmeans[:, 0], kmeans[:, 1],
                c=y, edgecolor='none', alpha=0.3,
                cmap=plt.cm.get_cmap("tab20", 10))
    plt.xlabel('Cluster 1')
    plt.ylabel('Cluster 2')
    # plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return


def plot_clust_col(kmeans, labs, n_clust, save_path, save_name):
    plt.figure(figsize=(7, 5))
    plt.scatter(kmeans[:, 0], kmeans[:, 1],
                c=labs.astype(float), edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap("tab20", n_clust))
    plt.xlabel('Cluster 1')
    plt.ylabel('Cluster 2')
    # ticks = list(range(1, n_clust + 1))
    # plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return


def find_K(X, save_path, save_name):
    sse = {}
    for k in range(1, 24):
        # ^ as 24 categories in kat_sport_a
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X)
        X["clusters"] = kmeans.labels_
        sse[k] = kmeans.inertia_
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return



def Plot_NMF_all(H, save_path, save_name):
    bounds = np.arange(start=0, stop=1.05, step=0.05)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    ticks = list(range(0, len(H.columns)))
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    img = ax.imshow(H, aspect='auto', interpolation='none', norm=norm)
    x_label_list = list(H.columns.values)
    ax.set_xticks(ticks)
    ax.set_xticklabels(x_label_list)
    fig.colorbar(img)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return



def plot_SHAP_force(i, X_test, model, save_path, save_name,
                    pred_model, title, figsize=(8, 4)):
    if (pred_model == "rf") or (pred_model == "tree"):
        explainerModel = shap.TreeExplainer(model)
    elif (pred_model == "enet") or (pred_model == "lasso"):
        masker = shap.maskers.Independent(data=X_test)
        explainerModel = shap.LinearExplainer(model, masker=masker)
    # todo: needs a mask
    else:
        print("please enter one of the regression or tree based models: 'rf', 'tree', 'lasso, or 'enet'")
        breakpoint()
    shap_values_Model = explainerModel.shap_values(X_test).round(2)
    plt.figure(figsize=figsize)
    p = shap.force_plot(explainerModel.expected_value.round(2), shap_values_Model[i],
                        round(X_test.iloc[[i]], 2), matplotlib=True, show=False)
    plt.gcf().set_size_inches(figsize)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path + save_name + '.png')
    plt.close()
    plt.clf()
    plt.cla()
    plt.close()
    return(p)

# todo: plot all 14 CMs
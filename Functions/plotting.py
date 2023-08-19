import seaborn as sns
import matplotlib.pyplot as plt
import colorcet as cc
import shap
import pandas as pd
import numpy as np
from matplotlib import colors
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from fixed_params import do_Enet

def plot_count(data, x, hue, xlabs, save_path, save_name, xlab, leg_labs,
               title=""):
    n = data[x].nunique() + 1
    # pal = sns.color_palette(cc.glasbey, n_colors=14)

    plt.figure(figsize=(7.8, 4))
    sns.histplot(data=data, x=x, hue=hue, stat="count", discrete=True, bins=14,
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
# todo: sort out ordering


def plot_perc(data, x, hue, xlabs, save_path, save_name, xlab, leg_labs, order, title=""):
    n = data[x].nunique() + 1
    plt.figure(figsize=(7.8, 4))
    sns.histplot(data=data, x=x, hue=hue, stat="percent", discrete=True, bins=14,
                 legend=False, hue_order=order)
    plt.xticks(ticks= range(1, n), labels=xlabs, rotation=60, size=7)
    plt.legend(bbox_to_anchor=(1, 1), title='Sport Category (B)', loc='upper left',
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


def plot_by_var(data, x, hue, xlabs, save_path, save_name, xlab, leg_labs, title=""):
    n = data[x].nunique() + 1
    plt.figure(figsize=(7.8, 4))
    sns.histplot(data=data, x=x, hue=hue, stat="count", discrete=True, bins=14,
                 legend=False)
    plt.xticks(ticks= range(1, n), labels=xlabs, rotation=60, size=7)
    plt.legend(bbox_to_anchor=(1, 1), title='Gender', loc='upper left',
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


def plot_impurity(impurity_imp_df, display_n, save_path, save_name):
    y_ticks = np.arange(0, display_n)
    fig, ax = plt.subplots()
    ax.barh(y_ticks, impurity_imp_df["Importance"])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(impurity_imp_df["Feature"])
    ax.set_title("Impurity Reduction Importance (training set)")
    fig.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    return


def plot_permutation(perm_imp_df, display_n, save_path, save_name):
    y_ticks = np.arange(0, display_n)
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
    plt.savefig(save_path + save_name)
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
    x_ticks = ["Dummy Most Frequent", "Dummy Random",
               "Dummy Stratified", "Logistic Regression",
               "Elastic Net", "Random Forest"]

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
    x_ticks = ["Dummy Most Frequent", "Dummy Random",
               "Dummy Stratified", "Logistic Regression",
               "Elastic Net", "Random Forest"]
    if do_Enet == False:
        x_ticks = x_ticks.remove("Elastic Net")

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



# todo: plot all 14 CMs
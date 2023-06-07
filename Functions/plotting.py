import seaborn as sns
import matplotlib.pyplot as plt
import colorcet as cc
import shap
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

def plot_count(data, x, hue, xlabs, save_path, save_name, xlab, leg_labs,
               title="", leg_title='Sport Category (B)'):
    n = data[x].nunique() + 1
    # pal = sns.color_palette(cc.glasbey, n_colors=14)

    plt.figure(figsize=(7.8, 4))
    sns.histplot(data=data, x=x, hue=hue, stat="count", discrete=True, bins=14,
                 legend=False)
    plt.xticks(ticks= range(1, n), labels=xlabs, rotation=60, size=7)
    plt.xlabel(xlab)
    plt.title(title)
    plt.subplots_adjust(right=0.3)
    plt.legend(bbox_to_anchor=(1, 1), title=leg_title, loc='upper left',
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
    palette = ["plum", "cornflowerblue", "coral", "mediumaquamarine"]

    sns.set_palette(palette)
    fig, ax = plt.subplots()
    plt.figure(figsize=(5, 4))

    data = data.transpose().reset_index()
    data.columns=data.iloc[0]
    data.drop(index=data.index[0], axis=0, inplace=True)
    data.rename(columns={'Unnamed: 0': 'Model'}, inplace=True)

    g=sns.barplot(x="Model", y=y, data=data, hue=colour, dodge=False,
                  palette=sns.color_palette(palette, 3))

    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)

    g.set_xticklabels(x_ticks)

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

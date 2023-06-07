import seaborn as sns
import matplotlib.pyplot as plt
import colorcet as cc

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
import matplotlib.pyplot as plt
import numpy as np
import os

# convergence accuracy
# {memory, latency, acc[mnli], acc[yelp-full]}
# ["roberta-large", "roberta-base", "bert-base", "albert-base"]
memory_latency_acc = {'agnews': [3.67, 5.817, 8, 5.817, 8],
 'mnli': np.array([1.0617, 1.977, 4.023, 1.977, 4.023])*2,
 'yahoo': [0.3923586347427407,
  0.384921039225675,
  0.3862455425369333,
  0.3998981151299032,
  0.774223127865512],
 'yelp-full': [0.42684, 0.4138, 0.4603, 0.49902, 0.57362]}

# convergence accuracy
# ['albert-base-v2', 'bert-base-uncased', 'bert-large-uncased', 'roberta-base', 'roberta-large']
acc = {'agnews': [0.85,
  0.8207894736842105,
  0.8610526315789474,
  0.8757894736842106,
  0.8925],
 'mnli': [0.3923586347427407,
  0.384921039225675,
  0.3862455425369333,
  0.3998981151299032,
  0.774223127865512],
 'yahoo': [0.6464833333333333,
  0.6367666666666667,
  0.6302833333333333,
  0.6098,
  0.6888666666666666],
 'yelp-full': [0.42684, 0.4138, 0.4603, 0.49902, 0.57362]}

fig_output_path = "../../figs"

def plot_models(dataset):
    """
        todo
    """
    label_font_conf = {
        # "weight": "bold",
        "size": "14"
    }
    bar_confs = {
        "color": ["white", "white", "silver", "grey", "black"],
        "linewidth": 1,
        "hatch": ["", "//", "", "//", ""],
        "edgecolor": "black",
    }


    figure_mosaic = """
    AAAAA.BBBBB.CCCCC.DDDDD
    """
    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(9, 2), dpi=100)
    bar_width = 0.03

    x = [0.1, 0.1+bar_width*2, 0.1+bar_width*4, 0.1+bar_width*6, 0.1+bar_width*8]

    xlabels = ["agnews", "mnli", "yahoo",'yelp-full']
    xlabels_fig = ["Jetson TX2", "Jetson TX2", "mnli",'yelp-full']
    ax = [axes["A"], axes["B"], axes["C"], axes["D"]]

    for i in range(len(axes)):
        ax[i].set_xlabel(xlabels_fig[i], **label_font_conf)
        ax[i].set_xticks([])
        fps = memory_latency_acc[xlabels[i]]
        ax[i].bar(x, fps, width=bar_width, **bar_confs)

        ax[i].grid(axis="y", alpha=0.3)
        ax[i].set_xlim(min(x)-bar_width*1.5, max(x)+bar_width*1.5)

    # ax[0].set_yscale('log')
    # ax[1].set_yscale('log')
    # ax[2].set_yscale('log')
    ax[0].set_ylabel("Memory (GB)", **label_font_conf)
    ax[1].set_ylabel("Latency (s)", **label_font_conf)
    ax[2].set_ylabel("Acc", **label_font_conf)
    # https://matplotlib.org/stable/api/container_api.html#module-matplotlib.container
    bars = ax[0].containers[0].get_children()
    labels = ['albert-base-v2', 'bert-base-uncased', 'bert-large-uncased', 'roberta-base', 'roberta-large']
    ax[0].legend(bars, labels, ncol=1, loc="lower left", bbox_to_anchor=(5.5, -0.1),frameon=False,fontsize=15,columnspacing = 1.5,handletextpad=0.5)

    plt.subplots_adjust(wspace=2.5)

    plt.savefig(os.path.join(fig_output_path, "different models.pdf"), bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    plot_models("agnews")

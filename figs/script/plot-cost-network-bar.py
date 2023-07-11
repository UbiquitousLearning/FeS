import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib as mpl


# per client unit: MB
# ["fecls","origin","bitfit", "bitfit+filter+curriculum"]

origin = {'agnews': [70000.0, 120.0, 70.0],
 'mnli': [230000.0, 140.0, 100.0],
 'yahoo': [160000.0, 160.0, 150.0],
 'yelp-full': [120000.0, 180.0, 40.0]}
 
network = {'agnews': np.array(origin['agnews']) / 1024,
 'mnli': np.array(origin['mnli']) / 1024,
 'yahoo': np.array(origin['yahoo']) / 1024,
 'yelp-full': np.array(origin['yelp-full']) / 1024}

reduction_1 = []
reduction_2 = []
for dataset in network:
    reduction_1.append(np.array(network[dataset][0]) / np.array(network[dataset][2]))
    reduction_2.append(np.array(network[dataset][1]) / np.array(network[dataset][2]))
print(reduction_1)
print(reduction_2)
print(np.mean(reduction_1))
print(np.mean(reduction_2))

print(network)

def plot_fps_stream_number(type):
    """
        plotting max stream numbers (online) or fps (offline) for each video
        type: 
            - online: plotting [libx264 (SoC-CPU), mediacodec(SoC-HW), libx264 (CPU), nvenc (GPU)]
            - offline: plotting [libx264 (SoC-CPU), libx264 (CPU), nvenc (GPU)]
    """
    label_font_conf = {
        # "weight": "bold",
        "size": "14"
    }
    bar_confs = {
        "color": ["white", "white", "silver"],
        "linewidth": 1,
        "hatch": ["", "//", ""],
        "edgecolor": "black",
    }

    figure_mosaic = """
    AAA.BBB.CCC.DDD
    """
    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(9, 2), dpi=100)
    bar_width = 0.03

    x = [0.1, 0.1+bar_width*2, 0.1+bar_width*4]
    data = network

    xlabels = ["agnews", "mnli", "yahoo", "yelp-full"]
    xlabels_fig = ["AGNEWS", "MNLI", "YAHOO", "YELP-F"]
    ax = [axes["A"], axes["B"], axes["C"], axes["D"]]

    for i in range(len(axes)):
        ax[i].set_xlabel(xlabels_fig[i], **label_font_conf)
        ax[i].set_xticks([])
        dataset = xlabels[i]  # video name
        fps = data[dataset]
        ax[i].bar(x, fps, width=bar_width, **bar_confs)

        ax[i].grid(axis="y", alpha=0.3)
        ax[i].set_xlim(min(x)-bar_width*1.5, max(x)+bar_width*1.5)

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')
    ax[3].set_yscale('log')
    ax[0].set_ylabel(r"Network Traffic (GB)", **label_font_conf)
    # https://matplotlib.org/stable/api/container_api.html#module-matplotlib.container
    bars = ax[0].containers[0].get_children()
    labels = ["FedFSL", "FedFSL-BIAS", "Ours"]
    ax[0].legend(bars, labels, ncol=3, loc="lower left", bbox_to_anchor=(0.8, 1),frameon=False,fontsize=15,columnspacing = 2,handletextpad=0.5)

    plt.subplots_adjust(wspace=2.5)
    # plt.show()
    plt.savefig('../../figs/cost/network.pdf', bbox_inches="tight")


if __name__ == '__main__':
    plot_fps_stream_number("offline")

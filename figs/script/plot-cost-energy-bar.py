import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib as mpl


# 5 clients
# ["fecls","origin","bitfit", "bitfit+filter+curriculum"]
origin = {'agnews': [7705000.0, 6881880.0, 382060.0],
 'mnli': [10896320.0, 2892980.0, 264280.0],
 'yahoo': [22139560.0, 10750600.0, 1247820.0],
 'yelp-full': [7855680.0, 6479640.0, 1017340.0]}
 
energy = {'agnews': np.array(origin['agnews']) / origin['agnews'][2],
 'mnli': np.array(origin['mnli']) / origin['mnli'][2],
 'yahoo': np.array(origin['yahoo']) / origin['yahoo'][2],
 'yelp-full': np.array(origin['yelp-full']) / origin['yelp-full'][2]}

reduction_1 = []
reduction_2 = []
for dataset in energy:
    reduction_1.append(np.array(energy[dataset][0]) / np.array(energy[dataset][2]))
    reduction_2.append(np.array(energy[dataset][1]) / np.array(energy[dataset][2]))
print(reduction_1)
print(reduction_2)
print(np.mean(reduction_1))
print(np.mean(reduction_2))

def plot_fps_stream_number(type):
    """
        plotting max stream numbers (online) or fps (offline) for each video
        type: 
            - online: plotting [libx264 (SoC-CPU), mediacodec(SoC-HW), libx264 (CPU), nvenc (GPU)]
            - offline: plotting [libx264 (SoC-CPU), libx264 (CPU), nvenc (GPU)]
    """
    label_font_conf = {
        # "weight": "bold",
        "size": "15"
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
    data = energy

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

        # # tag value on the last bar.
        # rect = ax[i].patches
        # height = rect[-1].get_height()
        # absolute_value = round(energy[dataset][-1] * 5 / 100, 1)
        # if i == 0:
        #     ax[i].text(x[-1]-0.025, height+0.2,absolute_value) 
        # elif i == 1:
        #     ax[i].text(x[-1]-0.025, height+0.5,absolute_value) 
        # elif i == 2:
        #     ax[i].text(x[-1]-0.025, height+0.30,absolute_value) 
        # elif i == 3:
        #     absolute_value = round(energy[dataset][-1] * 5 / 600, 1)
        #     ax[i].text(x[-1]-0.025, height+0.04,absolute_value) 

    ylabel = r'''Normalized Energy ($\times$)'''
    ax[0].set_ylabel(ylabel, **label_font_conf)
    # https://matplotlib.org/stable/api/container_api.html#module-matplotlib.container
    bars = ax[0].containers[0].get_children()
    # ["fecls","origin","bitfit", "bitfit+filter", "bitfit+filter+curriculum"]
    labels = ["FedFSL", "FedFSL-BIAS", "Ours"]
    ax[0].legend(bars, labels, ncol=3, loc="lower left", bbox_to_anchor=(0.8, 1),frameon=False,fontsize=15,columnspacing = 2,handletextpad=0.5)

    plt.subplots_adjust(wspace=2.5)
    # plt.show()
    plt.savefig('../../figs/cost/energy.pdf', bbox_inches="tight")


if __name__ == '__main__':
    plot_fps_stream_number("offline")
    print(energy)

import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib as mpl


# convergence runtime for each dataset, tx2, unit: min
conver_runtime_tx2 = {'agnews': np.array([43634.0, 3151.0, 2603.0, 751.0]) / 3600, 'mnli': np.array([87798.0, 4265.0, 3355.0, 1307.2]) / 3600, 'yahoo': np.array([10128.0, 1260.0, 362.0, 221.0]) / 3600, 'yelp-full': np.array([23308.0, 374.0, 211.5, 74.0]) / 3600}

# convergence runtime for each dataset, rpi
conver_runtime_rpi = {'agnews': [6074310.0, 5020810.0, 182520.0, 624217.5],
 'mnli': [188180.0, 3420330.0, 17900.0, 485385.0],
 'yahoo': [5442820.0, 3700610.0, 330805.0, 246250.0],
 'yelp-full': [1406750.0, 2094520.0, 293550.0, 535230.0]}

# 0.8 relative acc of full-set training, tx2
runtime_tx2 = {'agnews': [60520.0, 41720.0, 12336.0, 2191.0],
 'mnli': [5047.0],
 'yahoo': [46034.0, 24024.0, 8494.0, 2678.0],
 'yelp-full': [139952.0, 8058.0]}

# 0.8 relative acc of full-set training, rpi
runtime_rpi = {'agnews': [347800.0, 624820.0, 182520.0, 31885.0],
 'mnli': [72905.0],
 'yahoo': [270510.0, 359800.0, 126150.0, 39190.0],
 'yelp-full': [2094520.0, 119610.0]}

runtime_hybrid = {'agnews+TX2': [60520.0, 41720.0, 12336.0, 2191.0],
 'yahoo+TX2': [46034.0, 24024.0, 8494.0, 2678.0],
 'agnews+RPI': [347800.0, 624820.0, 182520.0, 31885.0],
 'yahoo+RPI': [270510.0, 359800.0, 126150.0, 39190.0]}

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
        "color": ["white", "white", "silver", "grey"],
        "linewidth": 1,
        "hatch": ["", "//", "", "//"],
        "edgecolor": "black",
    }

    figure_mosaic = """
    AAAA.BBBB.CCCC.DDDD
    """
    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(9, 2), dpi=100)
    bar_width = 0.03

    x = [0.1, 0.1+bar_width*2, 0.1+bar_width*4, 0.1+bar_width*6]
    data = conver_runtime_tx2

    xlabels = ['agnews', 'mnli', 'yahoo', 'yelp-full']
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
    ax[0].set_ylabel("Elapsed Training \nTime (hrs)", **label_font_conf)
    # https://matplotlib.org/stable/api/container_api.html#module-matplotlib.container
    bars = ax[0].containers[0].get_children()
    labels = ["FedFSL", "DC", "DC+RF", "Ours (DC+RF+CP)"]
    ax[0].legend(bars, labels, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1),frameon=False,fontsize=15,columnspacing = 1.0,handletextpad=0.3)

    plt.subplots_adjust(wspace=2.5)
    # plt.show()
    plt.savefig('../../figs/ablation/all/agnews-yahoo.pdf', bbox_inches="tight")


if __name__ == '__main__':
    plot_fps_stream_number("offline")

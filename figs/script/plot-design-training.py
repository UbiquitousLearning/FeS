import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib as mpl

with open("sys",'r') as f:
    sys=f.read()

datapoint = 32
latency = datapoint
bitfit_latency = latency * 0.5
freeze_latency = latency * 4 / 24

comm = 1000
bitfit_comm = comm * 0.001
freeze_comm = comm * 4 / 24

# [full, freeze, bitfit]
epoch_comm = {
    "tx2+high": np.array([comm, freeze_comm, bitfit_comm]) / 10 / 60,
    "tx2+low": np.array([comm, freeze_comm, bitfit_comm]) / 1 / 60,
    "rpi+high": np.array([comm, freeze_comm, bitfit_comm]) / 10 / 60,
    "rpi+low": np.array([comm, freeze_comm, bitfit_comm]) / 1 / 60
}

epoch_comp = {
    "tx2+high": np.array([latency, freeze_latency, bitfit_latency]) / 60 ,
    "tx2+low": np.array([latency, freeze_latency, bitfit_latency]) / 60,
    "rpi+high": np.array([latency, freeze_latency, bitfit_latency]) * 15 / 60,
    "rpi+low": np.array([latency, freeze_latency, bitfit_latency]) * 15 / 60
}

print(epoch_comm)
print(epoch_comp)
print("tx2+high", epoch_comm["tx2+high"] + epoch_comp["tx2+high"])
print("tx2+low", epoch_comm["tx2+low"] + epoch_comp["tx2+low"])
print("rpi+high", epoch_comm["rpi+high"] + epoch_comp["rpi+high"])
print("rpi+low", epoch_comm["rpi+low"] + epoch_comp["rpi+low"])
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
    bar_confs_comm = {
        "color": ["white", "silver", "grey"],
        "linewidth": 1,
        "hatch": ["x", "x", "x"],
        "edgecolor": "black",
    }

    bar_confs_comp = {
        "color": ["white", "silver", "grey"],
        "linewidth": 1,
        "hatch": ["o", "o", "o"],
        "edgecolor": "black",
    }

    figure_mosaic = """
    AAA.BBB.CCC.DDD
    """
    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(9, 2), dpi=100)
    bar_width = 0.03

    x = [0.1, 0.1+bar_width*2, 0.1+bar_width*4]
    data1 = epoch_comm
    data2 = epoch_comp

    xlabels = ["tx2+high", "tx2+low", "rpi+high", "rpi+low"]
    xlabels_fig = ["TX2+High", "TX2+Low", "RPI+High", "RPI+Low"]
    ax = [axes["A"], axes["B"], axes["C"], axes["D"]]

    for i in range(len(axes)):
        ax[i].set_xlabel(xlabels_fig[i], **label_font_conf)
        ax[i].set_xticks([])
        dataset = xlabels[i]  # video name
        comm = data1[dataset]
        comp = data2[dataset]
        ax[i].bar(x, comm, width=bar_width, **bar_confs_comm)

        bottom = comm

        ax[i].bar(x, comp, bottom=bottom, width=bar_width, **bar_confs_comp)


        ax[i].grid(axis="y", alpha=0.3)
        ax[i].set_xlim(min(x)-bar_width*1.5, max(x)+bar_width*1.5)

    # ax[0].set_yscale('log')
    # ax[1].set_yscale('log')
    # ax[2].set_yscale('log')
    # ax[3].set_yscale('log')
    ax[0].set_ylabel("Elapsed training time \nper epoch (min)", **label_font_conf)
    # https://matplotlib.org/stable/api/container_api.html#module-matplotlib.container
    bars = ax[0].containers[0].get_children()
    labels = ["Full", "Freeze", "Bias"]
    ax[0].legend(bars, labels, ncol=4, loc="lower left", bbox_to_anchor=(-0.6, 1),frameon=False,fontsize=15,columnspacing = 1.0,handletextpad=0.3)

    plt.subplots_adjust(wspace=2.5)
    # plt.show()
    plt.savefig('/Users/cdq/Desktop/opensource/FedPrompt/figs/bitfit/breakdown-bar.pdf', bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    plot_fps_stream_number("offline")

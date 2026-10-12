import matplotlib.pyplot as plt
import numpy as np



def export_figure_matplotlib(X, Y, Y_bin, label_names, f_name, basic_colors, dpi=300, resize_fact=1, plt_show=False):
    """
     Export array as figure in original resolution
     :param arr: array of image to save in original resolution
     :param f_name: name of file where to save figure
     :param resize_fact: resize facter wrt shape of arr, in (0, np.infty)
     :param dpi: dpi of your screen
     :param plt_show: show plot or not
     """


    fig = plt.figure()
    fig.set_size_inches(6, 6)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ids = np.arange(len(label_names))

    legend = False
    if legend:

        plt.rcParams['font.size']=12
        plt.rcParams["font.family"] = 'Times New Roman'
        if basic_colors != "none":
            for i, name in zip(ids, label_names):
                ax.scatter(X[Y == i, 0], X[Y == i, 1],
                           label=name, s=8, c=basic_colors[i])
        else:
            for i, name in zip(ids, label_names):
                ax.scatter(X[Y == i, 0], X[Y == i, 1],
                           label=name, s=8)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='lower center', ncol=8, markerscale=3., bbox_to_anchor=(0.5, -0.05), handlelength=1.2, handletextpad=0.3,
                      columnspacing=1.2)

        def export_legend(legend, filename="legend.png", expand=[-5, -5, 5, 5]):
            fig = legend.figure
            fig.canvas.draw()
            bbox = legend.get_window_extent()
            bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
            bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    else:
        if basic_colors != "none":
            for i, name in zip(ids, label_names):
                ax.scatter(X[Y == i, 0], X[Y == i, 1],
                           label=name, s=8, c=basic_colors[i])
        else:
            for i, name in zip(ids, label_names):
                ax.scatter(X[Y == i, 0], X[Y == i, 1],
                           label=name, s=8)

    plt.savefig(f_name, dpi=dpi)
    print(f_name)

    if plt_show:
        plt.show()
    else:
        plt.close()

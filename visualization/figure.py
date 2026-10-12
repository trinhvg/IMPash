import os

import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

random_state=42

class FigurePlot:
    """Root class to define a figure which cans be further plot using visdom
    """

    def __init__(self, title : str, ax=None, n_colors : int = 8):
        """Initializes a figure, setups figure; axes and canvas plus the colors

        Args:
            title (str): The title of the figure
            ax (Axes, optionnal): A default axes to plot. Default to None
            n_colors (int, optionnal): the number of colors to define. Default to 8
        """
        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.fig = self.ax.get_figure()

        self.title = title

        self.colors = [name for name, _ in mcolors.CSS4_COLORS.items()]
        basic_colors = ['c', 'm', 'r', 'g', 'b', 'y', 'k', 'orange']

        if n_colors <= len(basic_colors):
            self.colors = basic_colors

    def to_torch(self):
        """Given a figure, transforms the image into a Tensor

        Returns:
            Tensor: A Tensor containing the plot information
        """
        canvas = FigureCanvas(self.fig)
        canvas.draw()  # draw the canvas, cache the renderer
        image = torch.Tensor(np.array(
            self.fig.canvas.renderer.buffer_rgba()).transpose((2, 0, 1))
            )
        self.close()
        return image

    def set_figure_title(self):
        """Sets title to the figure
        """
        self.fig.suptitle(self.title)

    def close(self):
        """Closes the figure
        """
        plt.close(self.fig)

    def savefig(self, path : str):
        """Saves the figure under path

        Args:
            path (str): [description]
        """
        self.fig.savefig(path)


class BinaryMultimodalUmapPlot(FigurePlot):
    """For two domains data, plots 4 TSNEs/
        - TSNE with only domain 1 features and class labels
        - TSNE with only domain 2 features and class labels
        - TSNE with all features and class labels
        - TSNE with all features and domain labels
    """

    def __init__(self, X, Y, Y_bin, label_names, epoch, title, save_name, balance_domains=True, save_dir='viz_final_eccv'):
        """Creates a figure with the four plots

        Args:
            X (torch.Tensor): The features, N*D array
            Y (np.ndarray): The labels, N array
            Y_bin (np.ndarray): The domain labels, N array
            label_names (list): the names of the labels
            epoch (int): The epoch number
            title (str): The title of the figure
            balance_domains (bool, optionnal): Whether to plot same number of features from each domain. Default to True


        """
        import umap
        from sklearn.metrics import silhouette_score

        if balance_domains:
            #   We visualize same number of samples in both domains
            labels_source = Y[Y_bin==0]
            labels_target = Y[Y_bin==1]
            z_source = X[Y_bin==0]
            z_target = X[Y_bin==1]

            if len(labels_source) < len(labels_target):
                indexes = np.arange(len(labels_target))
                np.random.seed(random_state)
                np.random.shuffle(indexes)
                indexes = indexes[:len(labels_source)]
                labels_target = np.array(labels_target)[indexes].tolist()
                z_target = z_target[indexes]
            else:
                indexes = np.arange(len(labels_source))
                np.random.seed(random_state)
                np.random.shuffle(indexes)
                indexes = indexes[:len(labels_target)]
                labels_source = np.array(labels_source)[indexes].tolist()
                z_source = z_source[indexes]

            X = torch.cat([z_source, z_target])
            Y = np.concatenate((labels_source,labels_target))
            Y_bin = [0] * len(labels_source)
            Y_bin += [1] * len(labels_target)
            Y_bin = np.array(Y_bin)

        #   Get number of classes
        domain_ids = np.arange(1+max(Y_bin))
        n_colors = len(label_names)

        #   Setup the figure
        super().__init__(title, n_colors=n_colors)

        self.fig, self.ax = plt.subplots(2, 2, figsize=(12,10))

        reducer = umap.UMAP(random_state=42, n_neighbors=10)
        reducer.fit(X.numpy())
        X = reducer.transform(X.numpy())


        silhouette_scores = []
        silhouette_scores.append(silhouette_score(X, Y))
        for domain in range(5):
            silhouette_scores.append(silhouette_score(X[Y_bin==domain], Y[Y_bin==domain]))

        ids = np.arange(len(label_names))
        for i, name in zip(ids, label_names):
            silhouette_scores.append(silhouette_score(X[Y == i], Y_bin[Y == i]))


        from .save_fig import export_figure_matplotlib
        basic_colors = ['tab:blue', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray' ]
        basic_colors_2 = ['tab:olive', 'tab:cyan']

        export_figure_matplotlib(X, Y, Y_bin, label_names, f'{os.getcwd()}/{save_dir}/{save_name}0_class_labels.png', basic_colors='none')
        export_figure_matplotlib(X, Y_bin, Y_bin, [1, 2, 3, 4, 5, 6], f'{os.getcwd()}/{save_dir}/{save_name}1_domain_labels.png', basic_colors)

        import json
        log_file = f'{os.getcwd()}/{save_dir}/{save_name}0_silhouette_scores.json'
        json_data = {'silhouette': str(silhouette_scores)}
        with open(log_file, 'w') as json_file:
            json.dump(json_data, json_file)


        ids = np.arange(len(label_names))

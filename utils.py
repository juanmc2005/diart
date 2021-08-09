from pyannote.core import SlidingWindowFeature, notebook
import matplotlib.pyplot as plt


def visualize(feature: SlidingWindowFeature):
    notebook.crop = feature.extent
    plt.rcParams["figure.figsize"] = (8, 2)
    notebook.plot_feature(feature)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from utils import closest_factors


def plot_grid(images: np.ndarray, title: str = "", out_path: Optional[str] = None):
    r, c = closest_factors(len(images))
    fig, axs = plt.subplots(r, c, tight_layout=True, squeeze=False)
    for ax, img in zip(axs.ravel(), images):
        ax.imshow(img, cmap="gray")
        ax.axis(False)
    plt.suptitle(title)
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close(fig)

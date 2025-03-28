import matplotlib.pyplot as plt

from utils import closest_factors

from typing import Dict, List, Optional


def plot_scores(scores: Dict[str, List[float]], out_path: Optional[str] = None):
  r, c = closest_factors(len(scores))
  fig, axs = plt.subplots(r, c, tight_layout=True, figsize=(5 * c, 5 * r), squeeze=False)
  t = list(range(1, len(scores[list(scores.keys())[0]]) + 1))
  for i, key in enumerate(scores):
    ax = axs[i // c, i % c]
    ax.plot(t, scores[key])
    ax.axis(True)
    ax.grid(True)
    ax.set_xlabel("Steps")
    ax.set_ylabel(key)
    ax.set_title(f"{key} vs Steps")
    if "loss" in key.lower():
      max_score = min(scores[key])
    else:
      max_score = max(scores[key])
    max_index = scores[key].index(max_score)
    max_step = t[max_index]
    ax.annotate(
      f'Best: {max_score:.2f}',
      xy=(max_step, max_score),
      xytext=(max_step, max_score - (max_score * 0.05)),  # Slightly below
      arrowprops=dict(arrowstyle="->", color='red'),
      fontsize=10,
      color='red'
    )
  if out_path is not None:
    plt.savefig(out_path)
  else:
    plt.show()
  plt.close(fig)

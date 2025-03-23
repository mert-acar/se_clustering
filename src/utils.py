import os
import torch
import numpy as np
from shutil import rmtree
from random import shuffle

from typing import Union


def create_dir(output_path: str):
  if os.path.exists(output_path):
    c = input(f"Output path {output_path} is not empty! Do you want to delete the folder [y / n]: ")
    if "y" == c.lower():
      rmtree(output_path, ignore_errors=True)
    else:
      print("Exit!")
      raise SystemExit
  os.makedirs(output_path)


def get_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device("cuda")
  elif torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return torch.device("mps")
  else:
    return torch.device("cpu")


def select_random_samples(
  labels: Union[list[int], np.ndarray], num_samples_per_label: int, seed: int = 9001
) -> np.ndarray:
  unique_labels = np.unique(labels)
  selected_indices = []
  np.random.seed(seed)
  for label in unique_labels:
    indices = np.where(labels == label)[0]
    if len(indices) < num_samples_per_label:
      raise ValueError(f"Not enough samples for label {label}. Only {len(indices)} available.")
    selected = np.random.choice(indices, num_samples_per_label, replace=False)
    selected_indices.extend(selected)
  shuffle(selected_indices)
  return np.array(selected_indices)

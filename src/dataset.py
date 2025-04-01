import os
from typing import Optional, Callable, Tuple, List

import torch
import numpy as np
from PIL import Image
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader

DATASET_INFO = {
  "COIL20": {
    "num_classes": 20,
    "in_ch": 1
  },
  "CIFAR10": {
    "num_classes": 10,
    "in_ch": 3
  },
  "MNIST": {
    "num_classes": 10,
    "in_ch": 1
  },
}


class COIL20(Dataset):
  def __init__(self, data_root: str, split: Optional[str] = None, transform: Optional[Callable] = None):
    self.transform = transform

    if split is None:
      split = "all"

    self.filelist = [
      os.path.join(data_root, split, f)
      for f in os.listdir(os.path.join(data_root, split)) if f.endswith(".png")
    ]

  @property
  def targets(self) -> List[int]:
    return list(map(lambda x: int(x.split("/")[-1].split("__")[0].replace("obj", "")), self.filelist))

  def __len__(self) -> int:
    return len(self.filelist)

  def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
    fname = self.filelist[index]
    label = int(fname.split("/")[-1].split("__")[0].replace("obj", ""))
    img = Image.open(fname)

    if self.transform is not None:
      img = self.transform(img)
    return img, label


def flatten_tensor(x: torch.Tensor) -> torch.Tensor:
  return torch.flatten(x)


def get_transforms(dataset_name: str, split: str = "test") -> torch.nn.Module:
  dataset = dataset_name.lower()
  transform_list = [
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
  ]

  if split == "test":
    if "vector" in dataset:
      transform_list.append(transforms.Lambda(flatten_tensor))
    return transforms.Compose(transform_list)

  augmentation_list = []
  if dataset == "mnist":
    augmentation_list.append(
      transforms.RandomChoice([
        transforms.RandomAffine((-90, 90)),
        transforms.RandomAffine(0, translate=(0.2, 0.4)),  #type: ignore[reportArgumentType]
        transforms.RandomAffine(0, scale=(0.8, 1.1)),  #type: ignore[reportArgumentType]
        transforms.RandomAffine(0, shear=(-20, 20))  #type: ignore[reportArgumentType]
      ])
    )
  elif dataset == "cifar10":
    augmentation_list.append(transforms.RandomCrop(32, padding=8))
    augmentation_list.append(transforms.RandomHorizontalFlip())
    augmentation_list.append(transforms.RandomGrayscale(p=0.2))

  if "vector" in dataset:
    augmentation_list.append(transforms.Lambda(flatten_tensor))

  return transforms.Compose(transform_list + augmentation_list)


def get_dataset(dataset_name: str, split: str = "test", sample_per_class: Optional[int] = None) -> Dataset:
  dataset_name = dataset_name.lower()
  isTrain = split == "train"
  transform = get_transforms(dataset_name, split)
  if "mnist" in dataset_name:
    dataset = datasets.MNIST("../data/mnist/", train=isTrain, transform=transform, download=True)
  elif "cifar10" in dataset_name:
    dataset = datasets.CIFAR10("../data/cifar10/", train=isTrain, transform=transform, download=True)
  elif "coil20" in dataset_name:
    dataset = COIL20("../data/coil20/", split=split, transform=transform)
  else:
    raise NotImplementedError(f"{dataset_name} is not yet implemented")
  if sample_per_class is not None:
    dataset = sample_dataset(dataset, sample_per_class)
  return dataset


def get_dataloader(
  dataset_name: str,
  split: str,
  batch_size: int = 1,
  num_workers: int = 0,
  sample_per_class: Optional[int] = None,
) -> DataLoader:
  isTrain = split == "train"
  dataset_cls = get_dataset(dataset_name, split, sample_per_class)
  return DataLoader(dataset_cls, batch_size=batch_size, num_workers=num_workers, shuffle=isTrain)


def get_labels(dataset_name: str, split: str = "test") -> np.ndarray:
  if "mnist" in dataset_name.lower():
    dataset_name = "mnist"

  elif "cifar" in dataset_name.lower():
    dataset_name = "cifar"

  elif "coil20" in dataset_name.lower():
    dataset_name = "coil20"

  assert dataset_name.lower() in [
    "mnist", "cifar10", "coil20"
  ], f"dataset_name name must be one of ['coil20', 'mnist', 'cifar10'], got {dataset_name}"
  assert split in ["train", "test"], f"split must be one of ['train', 'test'], got {split}"

  return np.load(f"../data/{dataset_name.lower()}/{split}_labels.npy")


def sample_dataset(dataset: Dataset, num_samples_per_class: int) -> Dataset:
  labels = torch.tensor(dataset.targets)  # type: ignore
  classes = torch.unique(labels)
  sampled_indices = []
  for c in classes:
    class_indices = torch.where(labels == c)[0]
    perm = torch.randperm(len(class_indices))
    sampled_indices.extend(class_indices[perm[:num_samples_per_class]].tolist())
  return torch.utils.data.Subset(dataset, sampled_indices)

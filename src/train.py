import os
import torch
from tqdm import tqdm
from time import time
from pprint import pprint
from typing import Dict, Any
from yaml import full_load, dump

from visualize import plot_scores
from collections import defaultdict
from model import DSESCNet, AutoEncoder
from utils import get_device, create_dir
from dataset import get_dataloader, DATASET_INFO
from loss import SelfExpressiveLoss, ReconstructionLoss
from cluster import spectral_clustering, scores, silhouette_score
from scipy.sparse.linalg._eigen.arpack.arpack import ArpackError


def train_ae(config: Dict[str, Any], verbose: bool = True):
  if "output_path" not in config or config["output_path"] is None:
    config["output_path"] = f"../logs/AE_{config['data']['dataset_name'].upper()}/"
  create_dir(config["output_path"])
  ckpt_path = os.path.join(config["output_path"], f"best_state.pt")
  with open(os.path.join(config["output_path"], "ExperimentSummary.yaml"), "w") as f:
    dump(config, f)

  device = get_device()
  if verbose:
    print(f"[INFO] Running on {device}")

  dataset_info = DATASET_INFO[config["data"]["dataset_name"].upper()]
  config["data"]["sample_per_class"] = None
  dataloaders = {
    "train": get_dataloader(split="train", **config["data"]),
    "test": get_dataloader(split="test", **config["data"])
  }

  model = AutoEncoder(filters=config["model"]["filters"], in_ch=dataset_info["in_ch"]).to(device)

  optimizer = torch.optim.AdamW(
    model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"]
  )
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config["train"]["scheduler_args"])

  criterion = ReconstructionLoss()

  tick = time()
  best_epoch = -1
  phases = ["test", "train"]

  best_loss = float("inf")
  scores = defaultdict(list)
  for epoch in range(config["train"]["num_epochs"]):
    if verbose:
      print("-" * 20)
      print(f"Epoch {epoch + 1} / {config['train']['num_epochs']}")
    for phase in phases:
      if phase == "train":
        model.train()
      else:
        model.eval()
      running_loss = 0 
      if verbose:
        pbar = tqdm(
          dataloaders[phase], total=len(dataloaders[phase]), ncols=94
        )
      else:
        pbar = dataloaders[phase]

      with torch.set_grad_enabled(phase == "train"):
        for data, _ in pbar:
          data = data.to(device)
          output = model(data)
          loss = criterion(output, data)
          running_loss += loss.item()
          if phase == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

      running_loss = running_loss / len(dataloaders[phase])
      scores[f"{phase}_loss"].append(running_loss)
      if verbose:
        print(f"Loss: {running_loss:.6f}")
      if phase == "test":
        scheduler.step(running_loss)
        if running_loss < best_loss:
          best_loss = running_loss
          best_epoch = epoch
          if verbose:
            print(f"+ Saving the model to {ckpt_path}...")
            torch.save(model.state_dict(), ckpt_path)

    if epoch - best_epoch >= config["train"]["early_stop"]:
      if verbose:
        print(f"No improvements in {config['train']['early_stop']} epochs, stop!")
      break

  total_time = time() - tick
  m, s = divmod(total_time, 60)
  h, m = divmod(m, 60)
  if verbose:
    print(f"Training took {int(h):d} hours {int(m):d} minutes {s:.2f} seconds.")
    plot_scores(scores)

  config.update({"scores": {"loss": best_loss}})
  with open(os.path.join(config["output_path"], "ExperimentSummary.yaml"), "w") as f:
    dump(config, f)


def train_cluster(config: Dict[str, Any], verbose: bool = True):
  if "output_path" not in config or config["output_path"] is None:
    config["output_path"] = f"../logs/DSC_{config['data']['dataset_name'].upper()}/"
  create_dir(config["output_path"])
  ckpt_path = os.path.join(config["output_path"], f"best_state.pt")
  with open(os.path.join(config["output_path"], "ExperimentSummary.yaml"), "w") as f:
    dump(config, f)

  dataset_info = DATASET_INFO[config["data"]["dataset_name"].upper()]
  device = get_device()
  if verbose:
    print(f"[INFO] Running on {device}")

  config["data"]["batch_size"] = config["data"]["sample_per_class"] * dataset_info["num_classes"]
  x, y = next(iter(get_dataloader(split="train", **config["data"])))
  x = x.to(device)
  if min(y) > 0:
    y = y - min(y)
  y = y.numpy()

  model = DSESCNet(n_samples=len(y), filters=config["model"]["filters"], in_ch=dataset_info["in_ch"]).to(device)

  state_dict = torch.load(config['train']['pretrained_weights'], map_location=device)
  model.ae.load_state_dict(state_dict)

  for p in model.ae.parameters():
    p.requires_grad = False

  optimizer = torch.optim.Adam(
    model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"]
  )
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config["train"]["scheduler_args"])

  criterion = SelfExpressiveLoss(**config["train"]["criterion_args"])

  tick = time()
  best_epoch = -1
  best_acc = 0
  best_loss = float("inf")
  model.train()
  score_dict = defaultdict(list)
  for epoch in range(config["train"]["num_epochs"]):
    optimizer.zero_grad()
    x_recon, z, z_recon = model(x)
    loss = criterion(x, x_recon, z, z_recon, model.sec.c)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    if epoch % 5 == 0 or epoch == config["train"]["num_epochs"] - 1:
      coeff = model.sec.c.detach().cpu().numpy()
      try:
        y_pred = spectral_clustering(
          coeff=coeff,
          num_classes=dataset_info["num_classes"],
          dims=config["cluster"]["dims"],
          alpha=config["cluster"]["alpha"],
          ro=config["cluster"]["ro"]
        )
        cluster_scores = scores(y_pred, y)
        cluster_scores["silhouette_score"] = float(silhouette_score(z.detach().cpu().numpy(), y))
        for key in criterion.loss_dict:
          cluster_scores[key] = criterion.loss_dict[key]
        cluster_scores["loss"] = loss.item()

        for key in cluster_scores:
          score_dict[key].append(cluster_scores[key])

        if verbose:
          print("-" * 20)
          print(f"Epoch {epoch + 1} / {config["train"]['num_epochs']}")
          pprint(cluster_scores)

        if cluster_scores["accuracy"] > best_acc:
          best_acc = cluster_scores["accuracy"]
          if verbose:
            print(f"+ Saving the model to {ckpt_path}...")
          torch.save(model.state_dict(), ckpt_path)

      except ArpackError:
        pass

    if loss < best_loss:
      best_loss = loss
      best_epoch = epoch

    if epoch == 50:
      for p in model.ae.parameters():
        p.requires_grad = True
    if epoch - best_epoch >= config["train"]["early_stop"]:
      if verbose:
        print(f"No improvements in {config['train']['early_stop']} epochs, stop!")
      break

  total_time = time() - tick
  m, s = divmod(total_time, 60)
  h, m = divmod(m, 60)
  if verbose:
    print(f"Training took {int(h):d} hours {int(m):d} minutes {s:.2f} seconds.")
    plot_scores(score_dict)

  config.update({
    "scores": {
      f"best_{key}": max(value) if "loss" not in key else min(value)
      for key, value in score_dict.items()
    }
  })  
  with open(os.path.join(config["output_path"], "ExperimentSummary.yaml"), "w") as f:
    dump(config, f)
  return max(score_dict["accuracy"])


def main(train_type: str = "cluster", config_path: str = "./config.yaml",  verbose: bool = True):
  with open(config_path, "r") as f:
    config = full_load(f)
  if train_type == "cluster":
    train_cluster(config, verbose)
  elif train_type == "ae":
    train_ae(config, verbose)
  else:
    raise NotImplementedError(train_type)


if __name__ == "__main__":
  from fire import Fire
  Fire(main)

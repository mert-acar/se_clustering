import os
import torch
from tqdm import tqdm
from time import time
from pprint import pprint
from typing import Dict, Any
from yaml import full_load, dump

from model import DSESCNet, AutoEncoder
from loss import SelfExpressiveLoss, ReconstructionLoss
from cluster import spectral_clustering, scores
from utils import get_device, create_dir
from dataset import get_dataloader, DATASET_INFO


def train_ae(config: Dict[str, Any]):
  if "output_path" not in config or config["output_path"] is None:
    config["output_path"] = f"../logs/AE_{config['data']['dataset_name'].upper()}/"
  create_dir(config["output_path"])
  ckpt_path = os.path.join(config["output_path"], f"best_state.pt")
  with open(os.path.join(config["output_path"], "ExperimentSummary.yaml"), "w") as f:
    dump(config, f)

  device = get_device()
  print(f"[INFO] Running on {device}")

  dataset_info = DATASET_INFO[config["data"]["dataset_name"].upper()]
  config["data"]["sample_per_class"] = None
  dataloaders = {
    "train": get_dataloader(split="train", **config["data"]),
    "test": get_dataloader(split="test", **config["data"])
  }

  model = AutoEncoder(in_ch=dataset_info["in_ch"]).to(device)

  optimizer = torch.optim.AdamW(
    model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"]
  )
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config["train"]["scheduler_args"])

  criterion = ReconstructionLoss()

  tick = time()
  best_epoch = -1
  phases = ["test", "train"]

  best_loss = float("inf")
  model.train()
  for epoch in range(config["train"]["num_epochs"]):
    print("-" * 20)
    print(f"Epoch {epoch + 1} / {config["train"]['num_epochs']}")
    for phase in phases:
      if phase == "train":
        model.train()
      else:
        model.eval()
      running_loss = 0 
      with torch.set_grad_enabled(phase == "train"):
        for data, _ in tqdm(
          dataloaders[phase], total=len(dataloaders[phase]), ncols=94
        ):
          data = data.to(device)
          output = model(data)
          loss = criterion(output, data)
          running_loss += loss.item()
          if phase == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

      running_loss = running_loss / len(dataloaders[phase])
      print(f"Loss: {running_loss:.3f}")
      if phase == "test":
        scheduler.step(running_loss)
        if running_loss < best_loss:
          best_loss = running_loss
          best_epoch = epoch
          print(f"+ Saving the model to {ckpt_path}...")
          torch.save(model.state_dict(), ckpt_path)

    if epoch - best_epoch >= config["train"]["early_stop"]:
      print(f"No improvements in {config['train']['early_stop']} epochs, stop!")
      break

  total_time = time() - tick
  m, s = divmod(total_time, 60)
  h, m = divmod(m, 60)
  print(f"Training took {int(h):d} hours {int(m):d} minutes {s:.2f} seconds.")

  config.update({"scores": {"loss": best_loss}})
  with open(os.path.join(config["output_path"], "ExperimentSummary.yaml"), "w") as f:
    dump(config, f)



def train_model(config: Dict[str, Any]):
  if "output_path" not in config or config["output_path"] is None:
    config["output_path"] = f"../logs/DSESC_{config['data']['dataset_name'].upper()}/"
  create_dir(config["output_path"])
  ckpt_path = os.path.join(config["output_path"], f"best_state.pt")
  with open(os.path.join(config["output_path"], "ExperimentSummary.yaml"), "w") as f:
    dump(config, f)

  dataset_info = DATASET_INFO[config["data"]["dataset_name"].upper()]
  device = get_device()
  print(f"[INFO] Running on {device}")

  x, y = next(
    iter(
      get_dataloader(
        split="train", 
        batch_size=config["data"]["sample_per_class"] * dataset_info["num_classes"],
        **config["data"]
      )
    )
  )
  x, y = x.to(device), y.numpy() - 1

  model = DSESCNet(n_samples=len(y), in_ch=dataset_info["in_ch"]).to(device)
  state_dict = torch.load(
    f"../logs/AE_{config['data']['dataset_name'].upper()}/best_state.pt", map_location=device
  )
  model.load_state_dict(state_dict, strict=False)

  optimizer = torch.optim.AdamW(
    model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"]
  )
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config["train"]["scheduler_args"])

  criterion = SelfExpressiveLoss(**config["train"]["criterion_args"])

  tick = time()
  best_epoch = -1
  best_loss = float("inf")
  model.train()
  for epoch in range(config["train"]["num_epochs"]):
    print("-" * 20)
    print(f"Epoch {epoch + 1} / {config["train"]['num_epochs']}")
    x_recon, z, z_recon = model(x)
    loss = criterion(x, x_recon, z, z_recon, model.sec)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    if epoch % 10 == 0 or epoch == config["train"]["num_epochs"] - 1:
      coeff = model.sec.detach().cpu().numpy()
      y_pred = spectral_clustering(
        coeff=coeff,
        num_classes=dataset_info["num_classes"],
        dims=config["cluster"]["dims"],
        alpha=config["cluster"]["alpha"],
        ro=config["cluster"]["ro"]
      )
      cluster_scores = scores(coeff, y_pred, y)
      pprint(cluster_scores)

    if loss < best_loss:
      best_loss = loss
      best_epoch = epoch
      print(f"+ Saving the model to {ckpt_path}...")
      torch.save(model.state_dict(), ckpt_path)

    if epoch - best_epoch >= config["train"]["early_stop"]:
      print(f"No improvements in {config['train']['early_stop']} epochs, stop!")
      break

  total_time = time() - tick
  m, s = divmod(total_time, 60)
  h, m = divmod(m, 60)
  print(f"Training took {int(h):d} hours {int(m):d} minutes {s:.2f} seconds.")


def main(train_type: str = "cluster", config_path: str = "./config.yaml"):
  with open(config_path, "r") as f:
    config = full_load(f)
  if train_type == "cluster":
    train_model(config)
  elif train_type == "ae":
    train_ae(config)
  raise NotImplementedError(train_type)

if __name__ == "__main__":
  from fire import Fire
  Fire(main)

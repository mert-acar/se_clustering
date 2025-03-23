import os
import torch
from time import time
from typing import Dict, Any, Optional
from yaml import full_load, dump
from pprint import pprint

from model import DSESCNet
from loss import SelfExpressiveLoss
from cluster import spectral_clustering, scores
from utils import get_device, create_dir
from dataset import get_dataloader, DATASET_INFO


def train_model(config: Dict[str, Any], debug_every: Optional[int] = None):
  # Create the checkpoint output path
  create_dir(config["output_path"])
  ckpt_path = os.path.join(config["output_path"], f"best_state.pt")
  with open(os.path.join(config["output_path"], "ExperimentSummary.yaml"), "w") as f:
    dump(config, f)

  device = get_device()
  print(f"[INFO] Running on {device}")

  dataloader = get_dataloader(
    split="train", 
    batch_size=config["data"]["sample_per_class"] * DATASET_INFO[config["data"]["dataset_name"]]["num_classes"],
    **config["data"]
  )
  x, y = next(iter(dataloader))
  x, y = x.to(device), y.numpy()

  model = DSESCNet(len(y), in_ch=DATASET_INFO[config["data"]["dataset_name"]]["in_ch"]).to(device)
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
        num_classes=DATASET_INFO[config["data"]["dataset_name"]]["num_classes"],
        dims=12,
        alpha=8,
        ro=0.04
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


def main(config_path: str = "./config.yaml", debug_every: Optional[int] = None):
  with open(config_path, "r") as f:
    config = full_load(f)
  train_model(config, debug_every)


if __name__ == "__main__":
  from fire import Fire
  Fire(main)

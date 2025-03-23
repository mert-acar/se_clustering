import os
import torch
from tqdm import tqdm
from time import time
from typing import Dict, Any, Optional
from yaml import full_load, dump

from model import AutoEncoder
from dataset import get_dataloader
from loss import ReconstructionLoss
from utils import get_device, create_dir


def train_model(config: Dict[str, Any], debug_every: Optional[int] = None):
  # Create the checkpoint output path
  create_dir(config["output_path"])
  ckpt_path = os.path.join(config["output_path"], f"best_state.pt")
  with open(os.path.join(config["output_path"], "ExperimentSummary.yaml"), "w") as f:
    dump(config, f)

  device = get_device()
  print(f"[INFO] Running on {device}")

  dataloaders = {
    "train": get_dataloader(split="train", **config["data"]),
    "test": get_dataloader(split="test", **config["data"])
  }

  model = AutoEncoder().to(device)

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


def main(config_path: str = "./config.yaml", debug_every: Optional[int] = None):
  with open(config_path, "r") as f:
    config = full_load(f)
  train_model(config, debug_every)


if __name__ == "__main__":
  from fire import Fire
  Fire(main)

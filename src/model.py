import torch
import torch.nn as nn
from typing import Tuple


class AutoEncoder(nn.Module):
  def __init__(self, in_ch: int = 3) -> None:
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False),
      nn.ReLU(True),
      # nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
      # nn.ReLU(True),
      # nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
      # nn.ReLU(True),
    )

    self.decoder = nn.Sequential(
      # nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1),
      # nn.ReLU(True),
      # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1),
      # nn.ReLU(True),
      nn.ConvTranspose2d(32, in_ch, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1),
      nn.Sigmoid()
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.decoder(self.encoder(x))


class DSESCNet(nn.Module):
  def __init__(self, n_samples: int, in_ch: int = 3) -> None:
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False),
      nn.ReLU(True),
      # nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
      # nn.ReLU(True),
      # nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
      # nn.ReLU(True),
    )

    self.sec = nn.Parameter(
      1.0e-4 * torch.ones(n_samples, n_samples, dtype=torch.float32), requires_grad=True
    )
    self.decoder = nn.Sequential(
      # nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1),
      # nn.ReLU(True),
      # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1),
      # nn.ReLU(True),
      nn.ConvTranspose2d(32, in_ch, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1),
      nn.ReLU(True)
    )

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = self.encoder(x)
    shape = z.shape
    z = z.view(shape[0], -1)
    z_recon = torch.matmul(self.sec, z)
    x_recon = self.decoder(z_recon.view(shape))
    return x_recon, z, z_recon


if __name__ == "__main__":
  bs, in_ch, h, w = 2, 3, 32, 32
  x = torch.randn(bs, in_ch, h, w)
  model = DSESCNet(bs, in_ch)
  out, z, zc = model(x)
  assert out.shape == torch.Size([bs, in_ch, h, w]), "Input and output shapes do not match!"
  assert z.shape == torch.Size([bs, 32 * (h // 2) * (w // 2)]
                              ), "Latent code shape do not match with expectation!"
  assert zc.shape == torch.Size([bs, 32 * (h // 2) * (w // 2)]
                               ), "Reconstructed latent code shape do not match with expectation!"
  print("Shapes match ✅")

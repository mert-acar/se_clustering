import torch
import torch.nn as nn
from typing import Tuple, List


class AutoEncoder(nn.Module):
  def __init__(self, filters: List[int], in_ch: int = 3) -> None:
    super().__init__()
    channels = [in_ch] + filters
    self.encoder = nn.Sequential()
    for i, (in_ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
      self.encoder.add_module(f"conv{i + 1}", nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
      self.encoder.add_module(f"relu{i + 1}", nn.ReLU(inplace=True))

    channels = channels[::-1]
    self.decoder = nn.Sequential()
    for i, (in_ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
      self.decoder.add_module(
        f"deconv{i + 1}",
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
      )
      self.decoder.add_module(f"relu{i + 1}", nn.ReLU(inplace=True))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.decoder(self.encoder(x))


class SelfExpression(nn.Module):
  def __init__(self, n: int):
    super().__init__()
    self.c = nn.Parameter(1.0e-10 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    c_masked = self.c - torch.diag(torch.diag(self.c))
    y = torch.matmul(c_masked, x)
    return y


class DSESCNet(nn.Module):
  def __init__(self, n_samples: int, filters: List[int], in_ch: int = 3) -> None:
    super().__init__()
    self.ae = AutoEncoder(filters, in_ch)
    self.sec = SelfExpression(n_samples)

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = self.ae.encoder(x)
    shape = z.shape
    z = z.view(shape[0], -1)
    z_recon = self.sec(z)
    z_recon_reshaped = z_recon.view(shape)
    x_recon = self.ae.decoder(z_recon_reshaped)
    return x_recon, z, z_recon


if __name__ == "__main__":
  bs, in_ch, h, w = 2, 1, 32, 32
  filters = [16, 32, 64]
  x = torch.randn(bs, in_ch, h, w)
  model = DSESCNet(bs, filters, in_ch)
  print(model)
  out, z, zc = model(x)
  assert out.shape == torch.Size([bs, in_ch, h, w]), "Input and output shapes do not match!"
  assert z.shape == torch.Size([bs, filters[-1] * (h // (2**len(filters))) * (w // (2**len(filters)))]
                              ), "Latent code shape do not match with expectation!"
  assert zc.shape == torch.Size([bs, filters[-1] * (h // (2**len(filters))) * (w // (2**len(filters)))]
                               ), "Reconstructed latent code shape do not match with expectation!"
  print("Shapes match ✅")

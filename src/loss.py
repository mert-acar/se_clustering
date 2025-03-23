import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
  def forward(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, x_recon)


class SelfExpressiveLoss(nn.Module):
  def __init__(self, w_coeff: float = 1, w_self_exp: float = 1):
    super().__init__()
    self.w_coeff = w_coeff
    self.w_self_exp = w_self_exp
    self.frobenius = nn.MSELoss(reduction="sum")

  def forward(
    self, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor, z_recon: torch.Tensor, coeff: torch.Tensor
  ) -> torch.Tensor:
    auto_encoder_loss = self.frobenius(x_recon, x)
    coeff_loss = torch.sum(torch.pow(coeff, 2)) * self.w_coeff
    self_exp_loss = self.frobenius(z_recon, z) * self.w_self_exp
    loss = auto_encoder_loss + coeff_loss + self_exp_loss
    print(f"AE: {auto_encoder_loss:.3f}, C: {coeff_loss:.3f}, SE: {self_exp_loss:.3f}, Loss: {loss:.3f}")
    return loss

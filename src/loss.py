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
    self.loss_dict = {"ae_loss": 0.0, "c_loss": 0.0, "se_loss": 0.0}
    self.frobenius = nn.MSELoss(reduction="sum")

  def forward(
    self, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor, z_recon: torch.Tensor, coeff: torch.Tensor
  ) -> torch.Tensor:
    auto_encoder_loss = self.frobenius(x_recon, x)
    coeff_loss = torch.sum(torch.pow(coeff, 2)) * self.w_coeff
    self_exp_loss = self.frobenius(z_recon, z) * self.w_self_exp
    self._register_loss(auto_encoder_loss, coeff_loss, self_exp_loss)
    return auto_encoder_loss + coeff_loss + self_exp_loss

  def _register_loss(self, ae_loss: torch.Tensor, c_loss: torch.Tensor, se_loss: torch.Tensor):
    self.loss_dict["ae_loss"] = ae_loss.item()
    self.loss_dict["c_loss"] = c_loss.item()
    self.loss_dict["se_loss"] = se_loss.item()

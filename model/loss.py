import torch
import torch.nn as nn
import torch.nn.functional as F

# 三种损失函数有l1，l2和huber，默认为l1
class DiffusionLoss(nn.Module):
  def __init__(self, loss_type='l1') -> None:
     super().__init__()
     self.loss_type = loss_type
     
  def forward(self, noise, predicted_noise):
    if self.loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif self.loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif self.loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

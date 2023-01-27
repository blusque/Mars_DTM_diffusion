import torch
import torch.nn.functional as F
from schedule import *


class Coefficients:
    def __init__(self, num_steps, *, schedule=sigmoid_beta_schedule):
        # 制定每一步的beta
        self.betas = schedule(num_steps)

        # 计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * \
            (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        assert self.alphas.shape == self.alphas_cumprod.shape == self.alphas_cumprod_prev.shape\
            == self.sqrt_recip_alphas.shape == self.sqrt_alphas_cumprod.shape\
            == self.sqrt_one_minus_alphas_cumprod.shape == self.posterior_variance.shape  # 确保所有列表长度一致
        print("all the same shape: ", self.betas.shape)

    @staticmethod
    def extract(a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

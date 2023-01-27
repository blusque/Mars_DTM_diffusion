import model.index

from torchvision.utils import save_image
from torch.optim import Adam
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import DEMDataset
from model.coefficients import Coefficients
from model.model import Unet
from model.loss import DiffusionLoss
from plot import plot

import argparse
import os
import random
from pathlib import Path

parser = argparse.ArgumentParser(description="Pytorch MadNet 2.0 for mars")
parser.add_argument("--batch-size", type=int, default=8,
                    help="training batch size")
parser.add_argument("--nEpochs", type=int, default=100,
                    help="number of epochs to train for")
parser.add_argument("--num-steps", type=int, default=1000,
                    help="number of steps of noise and denoise")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="Generator Learning Rate. Default=1e-3")
parser.add_argument("--step", type=int, default=100,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str,
                    help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int,
                    help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1,
                    help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--beta1", default=0.9, type=float,
                    help="Adam beta 1, Default: 0.9")
parser.add_argument("--beta2", default=0.999, type=float,
                    help="Adam beta 2, Default: 0.999")
parser.add_argument("--weight-decay", "--wd", default=1e-4,
                    type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--save-per-epochs", "-p", default=10,
                    type=int, help="How many epochs the checkpoint is saved once.")
parser.add_argument("--dataset", "-d", default="",
                    type=str, help="Path to Dataset")

opt = parser.parse_args()
device = torch.device(
    "cuda" if opt.cuda and torch.cuda.is_available() else "cpu")
rse_data = []
ssim_data = []
epoch_data = []
num_steps = opt.num_steps
coeffs = Coefficients(num_steps)
extract = Coefficients.extract


def main():
    global opt
    print(opt)
    print(device)

    cuda = opt.cuda
    if os.name == "nt":
        opt.threads = 0
    if cuda and torch.cuda.is_available():
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    elif cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print("===> Loading datasets")
    transform = Compose([
        transforms.Lambda(lambda t: (t / 255. * 2) - 1)
    ])
    dataset = DEMDataset(opt.dataset, transform)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                            shuffle=True, drop_last=True, num_workers=opt.threads)

    print("===> Building Model")
    model = Unet(
        dim=8,
        dim_mults=(1, 2, 4)
    )
    model.to(device)
    if cuda:
        model = DataParallel(model)

    print("===> Setting Loss Functions")
    p_loss = DiffusionLoss(loss_type='huber')
    p_loss = p_loss.to(device)

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=>loading checkpoint {}".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'].state_dict())
        else:
            print("=> no checkpoint fount at {}".format(opt.resume))

    print("===> Setting Optimizer")
    betas = (opt.beta1, opt.beta2)
    optimizer = Adam(model.parameters(), lr=opt.lr, betas=betas)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.start_epoch + opt.nEpochs):
        train(epoch, model, dataloader, optimizer, p_loss)
        if epoch % opt.save_per_epochs == 0:
            save_checkpoint(model, epoch)


def q_x(x_0, t, noise=None):
    """可以基于x[0]得到任意时刻t的x[t]"""
    # x_0：dataset，原始(10000, 2)的数据点集
    # t: torch.tensor([i]),i为采样几次
    global coeffs
    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = extract(coeffs.sqrt_alphas_cumprod, t, x_0.shape)
    alphas_1_m_t = extract(coeffs.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return (alphas_t * x_0 + alphas_1_m_t * noise)  # 在x[0]的基础上添加噪声


@torch.no_grad()
def p_sample(model, x, t, t_index):
    # p_sample(model, img, torch.full((b,),i,device=device,dtype=torch.long),i)
    betas_t = extract(coeffs.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        coeffs.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(coeffs.sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * \
        (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:  # 加一定的噪声
        posterior_variance_t = extract(coeffs.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, shape):
    # 从噪声中逐步采样
    device = next(model.parameters()).device

    b = shape[0]

    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, num_steps)), desc='sampling loop time step', total=num_steps):
        img = p_sample(model, img, torch.full(
            (b, ), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


def adjust_learning_rate(epoch):
    global opt
    lr = opt.lr * \
        (0.1 ** ((epoch - opt.start_epoch + 1) // opt.step))
    return lr


def train(epoch, model, dataloader, optimizer, loss):
    p_loss = loss
    lr = adjust_learning_rate(epoch - 1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    print("Epoch={}, lr={}".format(epoch, lr))
    bar = tqdm(enumerate(dataloader), desc='Epoch {}/{}: '.format(epoch, opt.nEpochs),
               leave=False, total=len(dataloader))
    
    writer = SummaryWriter('./log')
    img_path = Path('./img')
    torch.autograd.set_detect_anomaly(True)
    
    for step, batch in bar:
        optimizer.zero_grad()  # 优化器数值清零

        dtm, ori = batch
        dtm = dtm.unsqueeze(1)
        ori = ori.unsqueeze(1)
        batch_size = dtm.shape[0]
        dtm = dtm.to(device)
        ori = ori.to(device)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, num_steps, (batch_size,),
                          device=device).long()  # 随机取t

        noise = torch.randn_like(dtm)
        x_t = q_x(dtm, t, noise)
        predicted_noise = model(x_t, ori, t)
        loss = p_loss(noise, predicted_noise)

        bar.set_postfix(Loss=loss.item())

        loss.backward()
        optimizer.step()
    pass


def save_checkpoint(model, epoch):
    model_folder = "./checkpoint/"
    model_out_path = model_folder + 'model_epoch_{}.pth'.format(epoch)
    model_state = {'epoch': epoch, 'model': model}

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(model_state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    main()
# betas：$\beta$

# alphas：$\alpha = 1 -\beta$

# alphas_cumprod：$\overline{\alpha_t} = \prod_{s = 1} ^ {t}\alpha_s$

# alphas_cumprod_prev: $\overline{\alpha_{t-1}}$

# sqrt_recip_alphas: $1 /\sqrt{\overline{\alpha_t}}$

# sqrt_alphas_cumprod: $\sqrt{\overline{\alpha_t}}$

# sqrt_one_minus_alphas_cumprod: $\sqrt{1 -\overline{\alpha_t}}$

# posterior_variance: $\beta * (1 -\overline{\alpha_{t-1}}) / (1 -\overline{\alpha_{t}})$

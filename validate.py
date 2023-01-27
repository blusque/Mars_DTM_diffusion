import torch.nn
from torch.nn import DataParallel
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from model.generator import Generator
from dataset import DEMDataset
from torch.utils.data import DataLoader

import sys
import os
sys.path.append('..')
from hill_shade import hill_shade
from line_drawer import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_rse(img0, img1):
    mse = (np.abs(img0 - img1) ** 2)
    return np.sqrt(mse).mean()


def matlab_style_gauss2d(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sum_h = h.sum()
    if sum_h != 0:
        h /= sum_h
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1.):
    if not im1.shape == im2.shape:
        raise ValueError("Input Images must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel, now you have {} channels.".format(len(im1.shape)))

    M, N = im1.shape
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    window = matlab_style_gauss2d(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigma_l2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma_l2 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    return np.mean(np.mean(ssim_map))


class Validator:
    def __init__(self, gt, predicted):
        self.gt = gt
        self.predicted = predicted

    def validate(self):
        channels = self.gt.shape[0]
        rse = 0.
        ssim = 0.
        for i in range(channels):
            gt = self.gt[i, 0, :, :]
            predicted = self.predicted[i, 0, :, :]
            rse += compute_rse(gt, predicted)
            ssim += compute_ssim(gt, predicted)
        rse /= channels
        ssim /= channels
        return rse, ssim


def show_result(ori, dtm, gen_dtm):
    for i in range(show_ori.shape[0]):
        gt = dtm[i, 0, :, :]
        ori = show_ori[i, 0, :, :]
        predicted = gen_dtm[i, 0, :, :]
        fig1, (ax11, ax12, ax13) = plt.subplots(1, 3)
        fig1.suptitle('origin')
        ax11.set_title('ground_truth')
        ax11.imshow(gt, cmap='gray')
        ax12.set_title('predicted')
        ax12.imshow(predicted, cmap='gray')
        ax13.set_title('ori')
        ax13.imshow(ori, cmap='gray')
        line_drawer_1 = LineDrawer(fig1, ax11)
        line_drawer_2 = LineDrawer(fig1, ax12)
        fig, ax = plt.subplots()
        def on_scroll(event):
            ax = event.inaxes
            y_min, y_max = ax.get_ylim()
            range_y = (y_max - y_min) / 10.
            if event.button == 'up':
                ax.set(ylim=(y_min + range_y, y_max - range_y))
            elif event.button == 'down':
                ax.set(ylim=(y_min - range_y, y_max + range_y))
            fig.canvas.draw_idle()  # 绘图动作实时反映在图像上
        fig.canvas.mpl_connect('scroll_event', on_scroll)
        line_drawer_1.draw((reshow_images, (gt, )), (show_profile, (fig, ax, 'ground_truth')))
        line_drawer_2.draw((reshow_images, (predicted, )), (show_profile, (fig, ax, 'prediected')))
        plt.show()
        gt_relief = hill_shade(gt, z_factor=10.0)
        predicted_relief = hill_shade(predicted, z_factor=10.0)
        predicted_relief = np.where(predicted_relief < 0.5, 0.5, predicted_relief)
        predicted_relief = np.where(predicted_relief > 0.9, 0.9, predicted_relief)
        fig2, (ax21, ax22) = plt.subplots(1, 2)
        fig2.suptitle('hill_shade_relief')
        ax21.set_title('gt_relief')
        ax21.imshow(gt_relief, cmap='gray')
        ax22.set_title('predicted_relief')
        ax22.imshow(predicted_relief, cmap='gray')
        plt.show()


def reshow_images(drawer, *images):
    for i in range(len(images)):
        im = images[i]
        drawer.ax.imshow(im, cmap='gray')


def show_profile(drawer, *args):
    if not drawer.line:
        return
    x = drawer.line[0].get_xdata()
    y = drawer.line[0].get_ydata()
    if drawer.mode == 'x':
        x = [0., 512.]
    elif drawer.mode == 'y':
        y = [0., 512.]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    t = np.arange(0, 1, 0.01, dtype=np.float32)
    x_sample = np.floor(x[0] + dx * t)
    y_sample = np.floor(y[0] + dy * t)
    x_sample = np.array(x_sample, dtype=np.int32)
    y_sample = np.array(y_sample, dtype=np.int32)
    img = drawer.ax.get_images()[0].get_array()
    if len(img.shape) == 3:
        img = img[:, :, 0]
    height = img[y_sample, x_sample]
    fig, ax, label = args
    ax.set_ylim((0., 1.))
    lines = ax.get_lines()
    if len(lines) == 2:
        ax.cla()
    lines = ax.plot(t, height, label=label)
    # fig.savefig('./validate_{}.png'.format(times))
    fig.legend()
    fig.show()
    # plt.close()

if __name__ == "__main__":
    io.use_plugin('matplotlib', 'imshow')
    model_path = '../checkpoint/model_epoch_1000.pth'
    gpus = [0]
    model = Generator().to(device)
    model = DataParallel(model, device_ids=gpus)
    state_dict = torch.load(model_path)['gen_model'].state_dict()
    model.load_state_dict(state_dict)
    dataset = None
    if os.name == "nt":
        dataset = DEMDataset("G:\\mini_dataset.hdf5")
    elif os.name == "posix":
        dataset = DEMDataset("/media/mei/Elements/mini_dataset.hdf5")
        # train_set = DEMDataset("../../data/mini_dataset_for_madnet2/mini_dataset.hdf5")
    batch_size = 1
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    err = 0.
    for iteration, batch in enumerate(data_loader, 1):
        dtm, ori = batch
        dtm = dtm / 255.
        ori = ori / 255.
        print('dtm max: {}, min: {}'.format(dtm.max(), dtm.min()))
        print('dtm mean: {}, var: {}'.format(dtm.mean(), dtm.var()))
        dtm = torch.unsqueeze(dtm, 1)
        ori = torch.unsqueeze(ori, 1)
        dtm = dtm.numpy()
        show_ori = ori.numpy()
        ori = ori.to(device)

        gen_dtm = model(ori).cpu().detach().numpy()
        # gen_dtm = 100 * np.log10(10 * gen_dtm)
        # gen_dtm /= gen_dtm.max()
        gen_dtm = np.abs(gen_dtm)
        gen_dtm = np.where(gen_dtm > 1., 1., gen_dtm)
        gen_dtm = np.where(gen_dtm < 0., 0., gen_dtm)
        # gen_dtm *= 254.
        for i in range(gen_dtm.shape[0]):
            print('gen_dtm {} max: {}, min: {}'.format(i, gen_dtm[i].max(), gen_dtm[i].min()))
            print('gen_dtm {} mean: {}, var: {}'.format(i, gen_dtm[i].mean(), gen_dtm[i].var()))

        val = Validator(dtm, gen_dtm)
        rse, ssim = val.validate()
        show_result(show_ori, dtm, gen_dtm)
        print("total: {}; now: {}".format(len(data_loader), iteration * batch_size))
        break

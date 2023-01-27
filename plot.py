# sphinx_gallery_thumbnail_path = "../../gallery/assets/transforms_thumbnail.png"
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T


plt.rcParams["savefig.bbox"] = 'tight'
orig_img = Image.open(Path('assets') / 'illustrate.jpg')
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)


def plot(imgs, with_orig=True, row_title=None, num_cols=6, drop_last=False, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        imgs = [imgs]

    imgs_len = 0
    for row in imgs:
        for img in row:
            imgs_len += 1
        
    num_rows = imgs_len // num_cols
    last = imgs_len % num_cols
    if last != 0:
        if not drop_last:
            num_rows += 1
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    end = False
    for row_idx, row in enumerate(imgs):
        if end:
            break
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            if drop_last and (row_idx + 1) * (col_idx + 1) == num_rows * num_cols:
                end = True
                break
            ax = axs[row_idx * col_idx // num_rows, row_idx * col_idx % num_rows]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot(None)

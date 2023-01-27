from time import time
import multiprocessing as mp
# import torch
# import torchvision
# from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import DEMDataset
import os
 
 
# transform = transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize((0.1307,), (0.3081,))
# ])

def check_workers(dataset_path):
    train_set = DEMDataset(dataset_path);
    print(f"num of CPU: {mp.cpu_count()}")
    for num_workers in range(2, mp.cpu_count(), 2):  
        train_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=64, pin_memory=True,
                                        shuffle=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

if os.name == 'nt':
    print("Running: ", __name__)
    if __name__ == "__main__":
        check_workers("G:/training_dataset.hdf5")
elif os.name == 'posix':
    # check_workers("/media/mei/Elements/training_dataset.hdf5")
    check_workers("/mnt/g/mini_dataset.hdf5")


# trainset = torchvision.datasets.MNIST(
#     root='dataset/',
#     train=True,  #如果为True，从 training.pt 创建数据，否则从 test.pt 创建数据。
#     download=True, #如果为true，则从 Internet 下载数据集并将其放在根目录中。 如果已下载数据集，则不会再次下载。
#     transform=transform
# )


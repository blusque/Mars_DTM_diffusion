from typing import Optional, Callable, Any
from torchvision.datasets import vision as vs
import torch
import h5py


class DEMDataset(vs.VisionDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.dtm_data, self.ori_data = self._load_data()
        
        assert self.dtm_data.shape == self.ori_data.shape
        print("All datasets have the same shape: ", self.ori_data.shape)

    def _load_data(self):
        file = h5py.File(self.root, 'r')
        dtm_data = file.get('dtm_grp/dst1')
        ori_data = file.get('ori_grp/dst1')
        return dtm_data, ori_data

    def __getitem__(self, index: int) -> Any:
        dtm = torch.from_numpy(self.dtm_data[index, ...]).float()
        ori = torch.from_numpy(self.ori_data[index, ...]).float()
        if self.transform is not None:
            dtm = self.transform(dtm)
        if self.target_transform is not None:
            ori = self.target_transform(ori)
        elif self.transform is not None:
            ori = self.transform(ori)
        return dtm, ori
            
    def __len__(self) -> int:
        return self.ori_data.shape[0]

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# class CustomDataset(Dataset):
#     def __init__(self, data_folder):
#         self.data_folder = data_folder
#         self.file_list = [f for f in os.listdir(data_folder) if f.endswith('.npy')]

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, index):
#         file_path = os.path.join(self.data_folder, self.file_list[index])
#         input_data = torch.from_numpy(np.load(file_path)).float()  # Ensure the input is of type float
#         target_data = torch.from_numpy(np.load(file_path.replace("input", "target"))).float()
#         return input_data, target_data

class CVPDataset(Dataset):
    def __init__(self, image_dir, gt_dir):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename = self.images[index]
        img_path = os.path.join(self.image_dir, self.images[index])
        gt_path = os.path.join(self.gt_dir, self.images[index])

        image = np.array(Image.open(img_path).convert("L"))
        image = image/255
        image = np.reshape(image,(1,) + image.shape)
        gt = np.array(Image.open(gt_path).convert("L"), dtype=np.float32)
        gt = gt/255
        gt = np.reshape(gt, (1,) + gt.shape)

        return filename, image, gt



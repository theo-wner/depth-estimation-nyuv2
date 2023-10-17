import os
import cv2
import random
import math
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.nn import functional as F
import pytorch_lightning as pl
import config
from utils import visualize_img_depth

'''
Defines classes for the NYUv2 dataset
'''

class NYUv2DataModule(pl.LightningDataModule):
    """
    Represents the NYUv2 DataModule needed for further simplification
    """
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    # Loads the dataset (not needed data already downloaded)
    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_dataset = NYUv2Dataset(split='train')
        self.val_dataset = NYUv2Dataset(split='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False)


class NYUv2Dataset(Dataset):
    """
    Represents the NYUv2 Dataset
    Example for obtaining an image: image, depth = dataset[0]
    """

    # Split can be either 'train' or 'test'
    def __init__(self, split='train'):
        self.root_dir = './data'
        self.split = split
        self.images_dir = os.path.join(self.root_dir, 'image', self.split)
        self.depths_dir = os.path.join(self.root_dir, 'depth', self.split)
        self.filenames = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.filenames)
    
    def _load_image(self, index):
        image_filename = os.path.join(self.images_dir, self.filenames[index])
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = TF.to_tensor(image)
        return image
    
    def _load_depth(self, index):
        depth_filename = os.path.join(self.depths_dir, self.filenames[index])
        depth = cv2.imread(depth_filename, cv2.IMREAD_GRAYSCALE)
        depth = torch.tensor(depth, dtype=torch.long).unsqueeze(0)
        return depth

    def __getitem__(self, index):
        image = self._load_image(index)
        depth = self._load_depth(index)

        # In case of training, apply data augmentation
        if self.split == 'train':
            # Randomly resize image and depth --> Image size changes!
            random_scaler = RandResize(scale=(0.5, 1.75))
            image, depth = random_scaler(image.unsqueeze(0).float(), depth.unsqueeze(0).float())

            # Random Horizontal Flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                depth = TF.hflip(depth)

            # Preprocessing for Random Crop
            if image.shape[1] < 480 or image.shape[2] < 640:
                height, width = image.shape[1], image.shape[2]
                pad_height = max(480 - height, 0)
                pad_width = max(640 - width, 0)
                pad_height_half = pad_height // 2
                pad_width_half = pad_width // 2
                border = (pad_width_half, pad_width - pad_width_half, pad_height_half, pad_height - pad_height_half)
                image = F.pad(image, border, 'constant', 0)
                depth = F.pad(depth, border, 'constant', config.IGNORE_INDEX)

            # Random Crop
            i, j, h, w = transforms.RandomCrop(size=(480, 640)).get_params(image, output_size=(480, 640))
            image = TF.crop(image, i, j, h, w)
            depth = TF.crop(depth, i, j, h, w)

        # In case of validation, do nothing
        elif self.split == 'test':
            pass

        return image, depth
    

class RandResize(object):
    """
    Randomly resize image & label with scale factor in [scale_min, scale_max]
    The size of the image gets changed!
    Source: https://github.com/Haochen-Wang409/U2PL/blob/main/u2pl/dataset/augmentation.py
    """
    def __init__(self, scale, aspect_ratio=None):
        self.scale = scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, image, label):
        if random.random() < 0.5:
            temp_scale = self.scale[0] + (1.0 - self.scale[0]) * random.random()
        else:
            temp_scale = 1.0 + (self.scale[1] - 1.0) * random.random()

        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = (self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random())
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)

        scale_factor_w = temp_scale * temp_aspect_ratio
        scale_factor_h = temp_scale / temp_aspect_ratio
        h, w = image.size()[-2:]
        new_w = int(w * scale_factor_w)
        new_h = int(h * scale_factor_h)
        image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
        label = F.interpolate(label, size=(new_h, new_w), mode="nearest")
        return image.squeeze(), label.squeeze(0)


if __name__ == '__main__':
    dataset = NYUv2Dataset(split='train')

    image, depth = dataset[7]

    print(depth.max())

    # Visualize Image and Depth Map
    visualize_img_depth(image, depth, depth, filename='test.png')

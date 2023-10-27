import os
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
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
        self.test_dataset = NYUv2Dataset(split='test')it='train')
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
        return image
    
    def _load_depth(self, index):
        depth_filename = os.path.join(self.depths_dir, self.filenames[index])
        depth = cv2.imread(depth_filename, cv2.IMREAD_GRAYSCALE)
        return depth
    
    def get_training_augmentation(self):
        train_augmentation = A.Compose([
            A.RandomScale(scale_limit=(-0.5, +0.75), p=1), # Relates to Scaling between 0.5 and 1.75
            A.PadIfNeeded(min_height=480, min_width=640, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), mask_value=0), # If the image gets smaller than 480x640    
            A.RandomCrop(height=480, width=640, p=1),
            A.HorizontalFlip(p=0.5),
            A.ToFloat(),
            ToTensorV2(transpose_mask=True)
        ])
        return train_augmentation
    
    def get_validation_augmentation(self):
        val_augmentation = A.Compose([
            A.ToFloat(),
            ToTensorV2(transpose_mask=True)
        ])
        return val_augmentation

    def __getitem__(self, index):
        image = self._load_image(index)
        depth = self._load_depth(index)

        # In case of training, apply data augmentation (ToTensor already included)
        if self.split == 'train':
            train_augmentation = self.get_training_augmentation()
            transformed = train_augmentation(image=image, mask=depth)
            image, depth = transformed['image'], transformed['mask'].unsqueeze(0)

        # In case of validation, apply validation augmentation (ToTensor already included)
        elif self.split == 'test':
            val_augmentation = self.get_validation_augmentation()
            transformed = val_augmentation(image=image, mask=depth)
            image, depth = transformed['image'], transformed['mask'].unsqueeze(0)
            
        return image, depth
    

if __name__ == '__main__':
    dataset = NYUv2Dataset(split='train')
    image, depth = dataset[0]
    print(image.shape)
    print(depth.shape)
    visualize_img_depth(image, depth, depth)
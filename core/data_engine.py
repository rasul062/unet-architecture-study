import sys
import os
sys.path.append("/content/dataset.py")
os.environ["PYTHONPATH"] = '/content/dataset.py'

from .dataset import VOCDataset  # Import your pre-loading dataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VOCDataEngine:
    def __init__(self, root_path, batch_size=16, num_workers=2, resolution=256):
        self.root_path = root_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        
        # Color Map is now internal - no more copy-pasting it!
        self.VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

    def get_transforms(self, train=True):
        """Internal logic for augmentations from your notebook."""
        if train:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def get_loaders(self):
        """Returns the ready-to-use train and val loaders."""
        train_ds = VOCDataset(
            mode='train', 
            root_path=self.root_path, 
            class_rgb_values=self.VOC_COLORMAP,
            resolution=self.resolution,
            augmentation=self.get_transforms(train=True)
        )
        val_ds = VOCDataset(
            mode='val', 
            root_path=self.root_path, 
            class_rgb_values=self.VOC_COLORMAP,
            resolution=self.resolution,
            augmentation=self.get_transforms(train=False)
        )

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        
        return train_loader, val_loader

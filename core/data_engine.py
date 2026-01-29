import sys
import os
sys.path.append("/content/dataset.py")
os.environ["PYTHONPATH"] = '/content/dataset.py'

from .dataset import VOCDataset  # Import your pre-loading dataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VOCDataEngine:
    """Engine for managing the PASCAL VOC data lifecycle, from augmentation to DataLoader creation."""
    def __init__(self, root_path, batch_size=16, num_workers=2, resolution=256):
        """Initializes data configurations and the standard VOC color mapping."""
        self.root_path = root_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        
        # Internal reference for PASCAL VOC RGB class mapping
        self.VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

    def get_transforms(self, train=True):
        """
        Defines image augmentation and normalization pipelines using Albumentations.
        
        The normalization constants (mean/std) are derived from the ImageNet dataset. 
        Using these specific values is standard practice when utilizing pre-trained 
        backbones (like ResNet), as it ensures the input distribution matches the 
        data the model was originally trained on.
        """
        if train:
            # Training pipeline: includes geometric and pixel-level augmentations
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        # Validation pipeline: consistent normalization without stochastic augmentation
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def get_loaders(self):
        """Instantiates VOCDataset objects and returns prepared PyTorch DataLoaders."""
        # Initialize the training subset
        train_ds = VOCDataset(
            mode='train', 
            root_path=self.root_path, 
            class_rgb_values=self.VOC_COLORMAP,
            resolution=self.resolution,
            augmentation=self.get_transforms(train=True)
        )

        # Initialize the validation subset
        val_ds = VOCDataset(
            mode='val', 
            root_path=self.root_path, 
            class_rgb_values=self.VOC_COLORMAP,
            resolution=self.resolution,
            augmentation=self.get_transforms(train=False)
        )

        # Construct loaders with hardware-optimized memory pinning
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        
        return train_loader, val_loader

import cv2
from torch.utils.data import Dataset
from tqdm import tqdm # Optional: nice progress bar for loading
import os
import numpy as np
import torch

class VOCDataset(Dataset):
    """
    Optimized PASCAL VOC2012 Dataset implementation.
    
    This class eliminates Disk I/O and CPU bottlenecks during training by pre-loading 
    and pre-encoding all images and masks into system RAM at startup.
    """
    def __init__(
        self,
        mode='train',
        root_path='/content/VOCdevkit/VOC2012_train_val/VOC2012_train_val',
        class_rgb_values=None,
        resolution = 256,
        augmentation=None,
        preprocessing=None,
      ):
        """Initializes dataset parameters and triggers the one-time RAM caching process."""
        self.class_rgb_values = class_rgb_values
        self.resolution = resolution
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mode = mode
        self.root_path = root_path

        # Establish directory structure for images, masks, and split files
        self.names_path = os.path.join(self.root_path, 'ImageSets', 'Segmentation', self.mode + '.txt')
        self.images_path = os.path.join(self.root_path, 'JPEGImages')
        self.masks_path = os.path.join(self.root_path, 'SegmentationClass')

        # Get valid filenames
        self.names = []
        self.read_names()

        # Load everything into RAM
        print(f"\n[Optimization] Pre-loading {len(self.names)} images and masks into RAM...")
        self.data = []

        # Using tqdm for a progress bar so you know it's working
        for name in tqdm(self.names):
            img_path = os.path.join(self.images_path, name + '.jpg')
            mask_path = os.path.join(self.masks_path, name + '.png')

            # Initial Disk Read: Convert BGR to RGB
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

            # Mask Encoding: Convert RGB values to integer class indices (done once at startup)
            mask = self.encode_mask(mask, self.class_rgb_values)

            # Store the processed tuple in memory
            self.data.append((image, mask))

        print("[Optimization] Loading complete. Training loop is now unblocked.\n")

    def __getitem__(self, i):
        """
        Retrieves a sample from RAM and applies on-the-fly resizing and augmentations.
        
        Note: Resizing is performed here to allow for dynamic resolution changes 
        without re-loading the entire RAM cache.
        """
        # Retrieve pre-loaded numpy arrays from RAM
        image, mask = self.data[i]

        # Resize Image -> Bilinear (Smooth)
        image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)

        # Use NEAREST neighbor interpolation for masks to prevent "ghost" class values
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Convert to Tensor
        if isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            mask = torch.from_numpy(mask).long()

        # Ensure image is tensor and handle cases where preprocessing might not return tensor
        if not isinstance(image, torch.Tensor):
             # Assuming channels last (H, W, C) -> (C, H, W)
             image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image, mask

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.names)

    def encode_mask(self, mask, class_rgb_values):
        """
        Performs semantic encoding by mapping RGB color codes to integer class IDs.
        
        Args:
            mask (np.ndarray): The 3-channel RGB mask from disk.
            class_rgb_values (list): List of RGB lists corresponding to VOC classes.
        Returns:
            np.ndarray: A single-channel integer semantic map.
        """
        semantic_map = np.zeros(mask.shape[:2], dtype=np.uint8)
        for class_id, color in enumerate(class_rgb_values):
            equality = np.equal(mask, color)
            class_map = np.all(equality, axis=-1)
            semantic_map[class_map] = class_id
        return semantic_map

    def read_names(self):
        """Parses the ImageSet text file and verifies file existence on disk."""
        f = open(self.names_path, 'r')
        lines = f.readlines()
        f.close()

        candidate_names = [line.strip() for line in lines if line.strip()]

        valid_names = []
        for name in candidate_names:
            img_p = os.path.join(self.images_path, name + '.jpg')
            mask_p = os.path.join(self.masks_path, name + '.png')
            if os.path.exists(img_p) and os.path.exists(mask_p):
                valid_names.append(name)

        self.names = valid_names

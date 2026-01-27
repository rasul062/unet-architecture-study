import cv2
from torch.utils.data import Dataset
from tqdm import tqdm # Optional: nice progress bar for loading
import os
import numpy as np
import torch

class VOCDataset(Dataset):
    """
    Optimized Pascal VOC2012 Dataset.
    Loads all images/masks into RAM at startup to eliminate Disk I/O and CPU bottlenecks.
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
        self.class_rgb_values = class_rgb_values
        self.resolution = resolution
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mode = mode
        self.root_path = root_path

        # Paths
        self.names_path = os.path.join(self.root_path, 'ImageSets', 'Segmentation', self.mode + '.txt')
        self.images_path = os.path.join(self.root_path, 'JPEGImages')
        self.masks_path = os.path.join(self.root_path, 'SegmentationClass')

        # 1. Get valid filenames
        self.names = []
        self.read_names()

        # 2. LOAD EVERYTHING INTO RAM
        print(f"\n[Optimization] Pre-loading {len(self.names)} images and masks into RAM...")
        self.data = []

        # Using tqdm for a progress bar so you know it's working
        for name in tqdm(self.names):
            img_path = os.path.join(self.images_path, name + '.jpg')
            mask_path = os.path.join(self.masks_path, name + '.png')

            # Read from Disk (The slow part - done ONCE)
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

            # Encode Mask (The REALLY slow part - done ONCE)
            mask = self.encode_mask(mask, self.class_rgb_values)

            # Store tuple in memory
            self.data.append((image, mask))

        print("[Optimization] Loading complete. Training loop is now unblocked.\n")

    def __getitem__(self, i):
        # 3. INSTANT ACCESS
        # Retrieve pre-loaded numpy arrays from RAM
        image, mask = self.data[i]

        # 2. MANUAL RESIZE (The Fix)
        # Resize Image -> Bilinear (Smooth)
        image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)

        # Resize Mask -> Nearest (Sharp, No Ghost Values)
        # CRITICAL: Ensure mask is integer before resizing to avoid float-interpolation
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)

        # Apply augmentations (Must be done on-the-fly for variety)
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

        # Ensure image is tensor (Handle cases where preprocessing might not return tensor)
        if not isinstance(image, torch.Tensor):
             # Assuming channels last (H, W, C) -> (C, H, W)
             image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image, mask

    def __len__(self):
        return len(self.names)

    def encode_mask(self, mask, class_rgb_values):
        # This function is unchanged, but now it only runs during __init__
        semantic_map = np.zeros(mask.shape[:2], dtype=np.uint8)
        for class_id, color in enumerate(class_rgb_values):
            equality = np.equal(mask, color)
            class_map = np.all(equality, axis=-1)
            semantic_map[class_map] = class_id
        return semantic_map

    def read_names(self):
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

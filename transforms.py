import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch

def get_train_transforms(mean, std):
    """Return training transformations"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.CoarseDropout(
            max_holes=1, 
            max_height=16, 
            max_width=16, 
            min_holes=1, 
            min_height=16,
            min_width=16,
            p=0.5
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

def get_test_transforms(mean, std):
    """Return test/validation transformations"""
    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

class AlbumentationsDataset:
    """Wrapper around dataset to apply albumentations transforms"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if self.transform:
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image = augmented['image']
            
        return image, label
    
    def __len__(self):
        return len(self.dataset)
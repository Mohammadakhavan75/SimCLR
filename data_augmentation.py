import torch
import torchvision.transforms as transforms
import random
from PIL import ImageFilter, ImageOps

class GaussianBlur:
    """Gaussian blur augmentation from SimCLR paper"""
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class SimCLRAugmentation:
    """SimCLR data augmentation pipeline - Paper Exact Configuration
    
    From paper: "We sequentially apply three simple augmentations: 
    (1) random cropping followed by resize back to the original size, 
    (2) random color distortions, and 
    (3) random Gaussian blur."
    """
    def __init__(self, size=224, dataset='imagenet', s=1.0):
        # Paper-specific configurations
        if dataset == 'cifar10':
            # CIFAR-10 specific settings from paper
            color_jitter = transforms.ColorJitter(0.8*0.5, 0.8*0.5, 0.8*0.5, 0.2*0.5)  # s=0.5 for CIFAR-10
            
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                # Note: No Gaussian blur for CIFAR-10 in paper
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                                   std=[0.2023, 0.1994, 0.2010])  # CIFAR-10 normalization
            ])
        else:
            # ImageNet settings from paper
            color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
            
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
    
    def __call__(self, x):
        # Generate two augmented views of the same image
        return self.transform(x), self.transform(x)
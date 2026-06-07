import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class AlbumentationsTransform:
    """
    Wrapper to use Albumentations with torchvision.datasets.ImageFolder.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        # ImageFolder provides PIL images, Albumentations needs numpy arrays
        img = np.array(img)
        augmented = self.transform(image=img)
        return augmented['image']

def get_dataloaders(data_dir, batch_size, model_specs):
    """
    Creates dataloaders with model-specific preprocessing.
    model_specs: dict containing 'input_size', 'mean', and 'std'
    """
    input_size = model_specs['input_size']
    mean = model_specs['mean']
    std = model_specs['std']

    # Training augmentation as requested by user
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(input_size, input_size), # 지름길 학습 방지
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # 조명 및 배경 바이어스 제거
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        
        # 잎의 변형 시뮬레이션
        A.OneOf([
            A.GridDistortion(p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
        ], p=0.3),
        
        # 특정 부위 과적합 방지 (정상 영역과 병반 영역 모두 골고루 보게 만듦)
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, 16), hole_width_range=(1, 16), p=0.5),
        
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # Basic transform for validation
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    try:
        train_dataset = datasets.ImageFolder(
            root=f"{data_dir}/train", 
            transform=AlbumentationsTransform(train_transform)
        )
        val_dataset = datasets.ImageFolder(
            root=f"{data_dir}/valid", 
            transform=val_transform
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, len(train_dataset.classes)
    except Exception as e:
        print(f"Warning: Could not load ImageFolder from {data_dir}. Error: {e}")
        return None, None, 10

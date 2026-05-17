import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size, model_specs):
    """
    Creates dataloaders with model-specific preprocessing.
    model_specs: dict containing 'input_size', 'mean', and 'std'
    """
    input_size = model_specs['input_size']
    mean = model_specs['mean']
    std = model_specs['std']

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    try:
        train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
        val_dataset = datasets.ImageFolder(root=f"{data_dir}/valid", transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, len(train_dataset.classes)
    except Exception as e:
        print(f"Warning: Could not load ImageFolder from {data_dir}. Error: {e}")
        return None, None, 10

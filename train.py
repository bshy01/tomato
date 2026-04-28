import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.model_factory import get_model
from src.datasets.data_loader import get_dataloaders
from src.utils.common import set_seed

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix(loss=running_loss/len(loader), acc=100.*correct/total)
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), 100. * correct / total

def run(config_path):
    # Load configs
    with open('configs/base_config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    with open(config_path, 'r') as f:
        exp_config = yaml.safe_load(f)
    
    config = {**base_config, **exp_config}
    set_seed(config.get('seed', 42))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model Specs & Data
    from src.models.model_factory import get_model_specs
    model_name = config['model']
    model_specs = get_model_specs(model_name)
    
    print(f"Applying specs for {model_name}: {model_specs}")

    # Data
    train_loader, val_loader, num_classes = get_dataloaders(
        config['data_path'], 
        config['batch_size'],
        model_specs=model_specs
    )
    
    if train_loader is None:
        print("Data loaders could not be initialized. Check your data directory.")
        return

    # Model
    model = get_model(config['model'], num_classes).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']))
    
    # Training Loop
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config file")
    args = parser.parse_args()
    run(args.config)

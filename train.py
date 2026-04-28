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

def run(args):
    # Load configs
    with open('configs/base_config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    with open(args.config, 'r') as f:
        exp_config = yaml.safe_load(f)
    
    # Merge configs
    config = {**base_config, **exp_config}
    
    # Override config with command-line arguments if provided
    if args.model:
        config['model'] = args.model
    if args.lr:
        config['learning_rate'] = args.lr
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    set_seed(config.get('seed', 42))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model Specs & Data
    from src.models.model_factory import get_model_specs
    model_name = config['model']
    model_specs = get_model_specs(model_name)
    
    print(f"--- Experiment Setup ---")
    print(f"Model: {model_name}")
    print(f"LR: {config['learning_rate']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Specs: {model_specs}")
    print(f"------------------------")

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
    parser = argparse.ArgumentParser(description="Tomato Leaf Classification Training")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config file")
    
    # Arguments to override config
    parser.add_argument("--model", type=str, choices=['resnet50', 'efficientnet_b0', 'vit'], help="Model architecture")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    
    args = parser.parse_args()
    run(args)

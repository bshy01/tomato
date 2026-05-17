import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support

from src.models.model_factory import get_model
from src.datasets.data_loader import get_dataloaders
from src.utils.common import set_seed

def calculate_metrics(all_labels, all_preds):
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    return precision, recall, f1

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
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
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        acc = 100. * (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
        pbar.set_postfix(loss=running_loss/len(loader), acc=acc)
    
    precision, recall, f1 = calculate_metrics(all_labels, all_preds)
    accuracy = 100. * (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
    
    return {
        'loss': running_loss / len(loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    precision, recall, f1 = calculate_metrics(all_labels, all_preds)
    accuracy = 100. * (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
    
    return {
        'loss': running_loss / len(loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

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
    
    # Setup Output and Logging
    exp_name = os.path.basename(args.config).split('.')[0]
    log_dir = os.path.join(config['output_path'], 'logs', exp_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    print(f"--- Experiment Setup ---")
    print(f"Model: {model_name}")
    print(f"LR: {config['learning_rate']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Logging to: {log_dir}")
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
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']))
    
    # Training Loop
    best_val_acc = 0.0
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Log to TensorBoard
        writer.add_scalars('Loss', {'Train': train_metrics['loss'], 'Val': val_metrics['loss']}, epoch)
        writer.add_scalars('Accuracy', {'Train': train_metrics['accuracy'], 'Val': val_metrics['accuracy']}, epoch)
        writer.add_scalars('Precision', {'Train': train_metrics['precision'], 'Val': val_metrics['precision']}, epoch)
        writer.add_scalars('Recall', {'Train': train_metrics['recall'], 'Val': val_metrics['recall']}, epoch)
        writer.add_scalars('F1-Score', {'Train': train_metrics['f1'], 'Val': val_metrics['f1']}, epoch)
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
        print(f"Val Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_path = os.path.join(config['output_path'], f"best_{exp_name}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    writer.close()

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

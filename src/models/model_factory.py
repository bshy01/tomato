import torch
import torch.nn as nn
import torchvision.models as models

# Centralized model specifications
MODEL_SPECS = {
    'resnet50': {
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'efficientnet_b0': {
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'vit': {
        'input_size': 224,
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]
    }
}

def get_model_specs(model_name):
    model_name = model_name.lower()
    if model_name not in MODEL_SPECS:
        print(f"Warning: {model_name} specs not found. Using defaults.")
        return MODEL_SPECS['resnet50']
    return MODEL_SPECS[model_name]

def get_model(model_name, num_classes, pretrained=True):
    model_name = model_name.lower()
    if model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'vit':
        model = models.vit_b_16(weights='DEFAULT' if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

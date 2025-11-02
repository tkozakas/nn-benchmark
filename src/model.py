from pathlib import Path

import torch
from torch import nn

from config import supported_architectures


def get_model(architecture, num_classes, pretrained=True):
    """Return a torchvision model instance for the given architecture.
    Supported architectures are defined in config.supported_architectures.

    Args:
        architecture: Model architecture name
        num_classes: Number of output classes
        pretrained: If True, loads ImageNet pretrained weights (default: True)
    """
    arch = architecture.lower()
    if architecture not in supported_architectures:
        raise ValueError(f"Unsupported architecture '{architecture}'. Choose one of: {list(supported_architectures.keys())}")

    if arch == 'googlenet':
        from torchvision.models import googlenet, GoogLeNet_Weights
        if pretrained:
            model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1, aux_logits=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model = googlenet(num_classes=num_classes, aux_logits=False)
        return model

    if arch == 'resnet18':
        from torchvision.models import resnet18, ResNet18_Weights
        if pretrained:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model = resnet18(num_classes=num_classes)
        return model

    if arch == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        if pretrained:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model = resnet50(num_classes=num_classes)
        return model

    if arch == 'densenet121':
        from torchvision.models import densenet121, DenseNet121_Weights
        if pretrained:
            model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        else:
            model = densenet121(num_classes=num_classes)
        return model

    raise ValueError(f"Architecture '{architecture}' not implemented.")


def save_model(model, path):
    project_root = Path(__file__).parent.parent.resolve()
    trained_dir = project_root / "trained"
    trained_dir.mkdir(parents=True, exist_ok=True)
    full_path = trained_dir / Path(path).name
    torch.save(model.state_dict(), full_path)
    print(f"Model saved to {full_path}")


def load_model(model, path):
    project_root = Path(__file__).parent.parent.resolve()
    trained_dir = project_root / "trained"
    full_path = trained_dir / Path(path).name
    model.load_state_dict(torch.load(full_path, map_location='cpu'))
    print(f"Model loaded from {full_path}")
    return model

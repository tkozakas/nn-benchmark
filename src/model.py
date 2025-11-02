from pathlib import Path

import torch

from config import supported_architectures


def get_model(architecture, num_classes):
    """Return a torchvision model instance for the given architecture.
    Supported architectures are defined in config.supported_architectures.
    """
    arch = architecture.lower()
    if architecture not in supported_architectures:
        raise ValueError(f"Unsupported architecture '{architecture}'. Choose one of: {list(supported_architectures.keys())}")
    if arch == 'googlenet':
        from torchvision.models import googlenet
        return googlenet(num_classes=num_classes, aux_logits=False)
    if arch == 'resnet18':
        from torchvision.models import resnet18
        return resnet18(num_classes=num_classes)
    if arch == 'resnet50':
        from torchvision.models import resnet50
        return resnet50(num_classes=num_classes)
    if arch == 'densenet121':
        from torchvision.models import densenet121
        return densenet121(num_classes=num_classes)
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

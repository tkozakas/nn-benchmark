import torch
from torchvision import transforms


def get_transforms(augment=True):
    """Get transforms for TinyImageNet (64x64 images).

    Args:
        augment: If True, applies strong data augmentation for training
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])


def get_subsample(full_dataset, subsample_size):
    """Optionally subsample the dataset."""
    if subsample_size:
        full_dataset, _ = torch.utils.data.random_split(
            full_dataset,
            [subsample_size, len(full_dataset) - subsample_size]
        )
    return full_dataset


def parse_args(args):
    """Parse command line arguments (TinyImageNet version)."""
    ARCHITECTURE = args['--architecture']
    N = int(args['--epochs'])
    B = int(args['--batch-size'])
    LR = float(args['--lr'])
    WD = float(args['--weight-decay'])
    CPU_WORKERS = int(args['--cpu-workers'])
    DEVICE = args['--device']

    K = None if args['--k-folds'] is None or (str(args['--k-folds']).isdigit() and int(args['--k-folds']) < 2) or str(
        args['--k-folds']).lower() == 'none' else int(args['--k-folds'])
    PAT = None if args['--patience'] is None or str(args['--patience']).lower() == 'none' else int(args['--patience'])
    SUBSAMPLE_SIZE = None if args['--subsample-size'] is None or str(args['--subsample-size']).lower() == 'none' else int(
        args['--subsample-size'])

    print(f"Device:       {DEVICE}")
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Subsample:    {SUBSAMPLE_SIZE!r}")

    print(f"Config → TinyImageNet, "
          f"Subsample size: {SUBSAMPLE_SIZE}, "
          f"Using K={K} folds, "
          f"N={N} epochs, "
          f"B={B} batch size, "
          f"LR={LR} learning rate, "
          f"WD={WD} weight decay, "
          f"PAT={PAT} patience, "
          f"CPU_WORKERS={CPU_WORKERS} workers")

    return (
        ARCHITECTURE,
        B,
        CPU_WORKERS,
        DEVICE,
        K,
        LR,
        N,
        PAT,
        SUBSAMPLE_SIZE,
        WD
    )

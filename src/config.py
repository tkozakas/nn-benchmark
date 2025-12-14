"""Configuration constants for the benchmark."""

# Training configuration
VAL_SPLIT = 0.2
RANDOM_STATE = 42

# Dataset configuration
DATASET_NAME = "akash2sharma/tiny-imagenet"
DATASET_SUBDIR = "tiny-imagenet-200/train"
IMAGE_SIZE = 64
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Supported model architectures
SUPPORTED_ARCHITECTURES = {
    'GoogleNet': {},
    'ResNet18': {},
    'ResNet50': {},
    'DenseNet121': {},
}

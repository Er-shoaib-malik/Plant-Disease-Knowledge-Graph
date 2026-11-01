from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def train_transforms(size: int = 224):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def inference_transforms(size: int = 224):
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size),
                               transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

from pathlib import Path
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from .preprocessing import train_transforms, inference_transforms

class ValidImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform, is_valid_file=self.valid_image)
    @staticmethod
    def valid_image(path):
        try:
            from PIL import Image
            with Image.open(path) as im: im.verify()
            return True
        except Exception:
            return False

def load_datasets(root: str | Path, image_size=224, seed=42):
    root = Path(root)
    if not root.is_dir(): raise FileNotFoundError(f"Dataset directory not found: {root}. See README for PlantVillage setup.")
    indexed = ValidImageFolder(root, transform=inference_transforms(image_size))
    if len(indexed) < 3: raise ValueError("Dataset must contain at least three valid images.")
    y = [label for _, label in indexed.samples]
    indices = list(range(len(y)))
    train_idx, hold_idx = train_test_split(indices, test_size=.2, random_state=seed, stratify=y)
    hold_y = [y[i] for i in hold_idx]
    val_idx, test_idx = train_test_split(hold_idx, test_size=.5, random_state=seed, stratify=hold_y)
    train_data = ValidImageFolder(root, transform=train_transforms(image_size))
    return Subset(train_data, train_idx), Subset(indexed, val_idx), Subset(indexed, test_idx), indexed.classes

from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainConfig:
    data_dir: Path = Path("ml/data/PlantVillage")
    output_dir: Path = Path("ml/artifacts")
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 1e-4
    num_workers: int = 4
    seed: int = 42

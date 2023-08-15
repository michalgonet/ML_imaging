import json
from dataclasses import dataclass
from enum import Enum, auto


@dataclass(frozen=True)
class Config:
    image_size: int
    n_channels: int
    batch_size: int
    epochs: int
    train_split: float
    val_split: float
    n_classes: int
    shuffle: bool
    shuffle_size: int
    seed: int
    resizing: bool
    rescaling: bool
    random_flip: bool
    random_rotation: float
    optimizer: str
    learning_rate: float
    loss_function: str
    metrics: list[str]

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    images_path: str
    labels_path: str
    image_width: int
    image_height: int
    image_channel: int
    n_classes: int
    epochs: int
    batch_size: int
    shuffle: bool
    transfer_learning: bool
    transfer_network: str
    test_size: float
    optimizer: str
    loss_function: str
    metrics: list[str]

from typing import Dict, Any


class TrainConfig:
    def __init__(self, optimizer: str, optimizer_params: Dict[str, Any], scheduler: str, scheduler_params: Dict[str, Any],
                 batch_size: int, train_ratio: float, valid_ratio: float, epochs: int) -> None:
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.epochs = epochs
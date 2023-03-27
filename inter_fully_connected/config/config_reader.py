from inter_fully_connected.config.train_config import TrainConfig
from yaml import load


class ConfigReader:
    @staticmethod
    def read_train_config_from_yaml_file(config_file: str) -> TrainConfig:
        with open(config_file, mode='r') as f:
            config = load(f)

        assert config["train_ratio"] + config["valid_ratio"] <= 1.0
        
        return TrainConfig(
            optimizer=config["optimizer"],
            optimizer_params=config["optimizer_params"],
            scheduler=config["scheduler"],
            scheduler_params=config["scheduler_params"],
            batch_size=config["batch_size"],
            train_ratio=config["train_ratio"],
            valid_ratio=config["valid_ratio"],
            epochs=config["epochs"]
        )
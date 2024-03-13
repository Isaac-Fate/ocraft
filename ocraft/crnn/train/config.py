from pathlib import Path
from pydantic import BaseModel


class TrainingConfig(BaseModel):

    # Root directory of the run,
    # which stores the log file and all checkpoints
    run_dir: Path

    # Root directory of the dataset
    dataset_dir: Path

    # Path to the file consisting of all tokens
    tokens_file_path: Path

    # Batch sizes
    train_batch_size: int
    valid_batch_size: int

    # Model configurations
    in_channels: int
    hidden_size: int
    image_tensor_height: int
    image_tensor_width: int

    # Training configurations
    epochs: int
    adam_lr: float
    exponential_lr_gamma: float

    # Save a checkpoint every n epochs
    # The checkpoint of the last epochs is always saved
    save_every_n_epochs: int

    # Whether to enable wandb
    wandb: bool = False

    @property
    def run_name(self) -> str:
        """The name of the run is given by
        the stem of the run directory.
        """

        return self.run_dir.stem

    @property
    def log_file_path(self) -> Path:
        """File path of the training log."""

        return self.run_dir.joinpath("train").with_suffix(".log")

    @property
    def train_samples_file_path(self) -> list[str]:
        """Path of the training sample annotation CSV."""

        return self.run_dir.joinpath("train-samples.csv")

    @property
    def valid_samples_file_path(self) -> list[str]:
        """Path of the validation sample annotation CSV."""

        return self.run_dir.joinpath("valid-samples.csv")

    @property
    def checkpoints_dir(self) -> Path:
        """Directory storing all checkpoints."""

        path = self.run_dir.joinpath("checkpoints")

        # Create the dir if it does not exist
        if not path.is_dir():
            path.mkdir()

        return path

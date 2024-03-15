from typing import Optional, Any
from pathlib import Path
from pydantic import BaseModel


class TrainingConfig(BaseModel):

    # Root directory of the run,
    # which stores the log file and all checkpoints
    run_dir: Path

    # Root directory of the dataset
    dataset_dir: Path

    # Path to the file consisting of all tokens
    # If this is None, the file path will be <run_dir>/tokens.txt
    tokens_file_path: Optional[Path] = None

    # Path of the training sample annotation CSV
    # If this is None, the file path will be <run_dir>/train-samples.csv
    train_samples_file_path: Optional[Path] = None

    # Path of the validation sample annotation CSV
    # If this is None, the file path will be <run_dir>/valid-samples.csv
    valid_samples_file_path: Optional[Path] = None

    # Batch sizes
    train_batch_size: int
    valid_batch_size: int

    # Model configurations
    in_channels: int
    hidden_size: int
    image_tensor_height: int
    image_tensor_width: int

    # Number of epochs to train
    epochs: int

    # Learning rate of the optimizer
    lr: float

    # The parameter of the exponential learning rate scheduler
    # The learning rate is multiplied by this parameter every epoch
    exponential_lr_gamma: float

    # Maximum gradient norm
    max_grad_norm: float

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
    def checkpoints_dir(self) -> Path:
        """Directory storing all checkpoints."""

        path = self.run_dir.joinpath("checkpoints")

        # Create the dir if it does not exist
        if not path.is_dir():
            path.mkdir()

        return path

    def model_post_init(self, __context: Any) -> None:

        # Call method of super class
        super().model_post_init(__context)

        # Infer the path of the tokens file
        if self.tokens_file_path is None:
            self.tokens_file_path = self.run_dir.joinpath("tokens.txt")

        # Infer the path of the training sample annotation CSV
        if self.train_samples_file_path is None:

            file_path = self.run_dir.joinpath("train-samples.csv")

            # Ensure that the file exists
            assert file_path.is_file(), f"File {file_path} does not exist"

            self.train_samples_file_path = self.run_dir.joinpath("train-samples.csv")

        # Infer the path of the validation sample annotation CSV
        if self.valid_samples_file_path is None:

            file_path = self.run_dir.joinpath("valid-samples.csv")

            # Ensure that the file exists
            assert file_path.is_file(), f"File {file_path} does not exist"

            self.valid_samples_file_path = self.run_dir.joinpath("valid-samples.csv")

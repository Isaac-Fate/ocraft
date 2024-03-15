from typing import Self, Optional
import tomllib
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR

import wandb
from loguru import logger

from .config import TrainingConfig
from .logging import BatchTrainingLogRecord, EpochTrainingLogRecord, ValidationLogRecord
from ..data import SynthDataset, ImageConverter, TextConverter, SynthSample
from ..models import CRNN


PROJECT_NAME = "crnn"


class Trainer:

    def __init__(self, config: TrainingConfig) -> None:

        self._config = config

        # Set up logger
        self._logger = logger.bind(key=PROJECT_NAME)
        self._logger.add(self._config.log_file_path)

        # Check if CUDA (GPU) is available, else use CPU
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data

        # Datasets and loaders will be initialized later in `_prepare_data`
        self._dataset: Optional[SynthDataset] = None
        self._train_data_loader: Optional[DataLoader] = None
        self._valid_data_loader: Optional[DataLoader] = None

        # Image converter
        self._image_converter = ImageConverter(
            self._config.image_tensor_height,
            self._config.image_tensor_width,
        )

        # Text converter will be initialized later in `_prepare_data`
        # since we need to read the file of tokens
        self._text_converter: Optional[TextConverter] = None

        # Model will be initialized later in `_prepare_data`
        self._model: Optional[CRNN] = None

    @property
    def device(self) -> str:
        """Device to be used."""

        return self._device

    @property
    def model(self) -> Optional[CRNN]:
        """The model to train."""

        return self._model

    @classmethod
    def from_config_file(cls, file_path: Path | str) -> Self:

        # Get the run dir
        run_dir = file_path.parent

        # Read the configuration file
        with open(file_path, "rb") as f:
            data = tomllib.load(f)

        # Set run dir
        data["run_dir"] = run_dir

        # Convert to TrainingConfig
        config = TrainingConfig.model_validate(data)

        return cls(config)

    def train(self):

        if self._config.wandb:

            # Log in to Wandb
            wandb.login()

            # Create a run
            run = wandb.init(
                # Project name
                project=PROJECT_NAME,
                # Run name
                name=self._config.run_name,
            )

        # Prepare data
        self._prepare_data()

        # Optimizer
        optimizer = RMSprop(
            self.model.parameters(),
            lr=self._config.lr,
        )

        # Scheduler
        # It is used to decrease the learning rate
        scheduler = ExponentialLR(
            optimizer,
            gamma=self._config.exponential_lr_gamma,
        )

        # Log
        self._logger.info("Start training...")
        self._logger.info(f"Training Configuration: {self._config}")

        # Training loop
        for i in range(self._config.epochs):

            # Epoch number
            epoch = i + 1

            # Training losse of each batch
            train_losses = []

            # Total number of batches
            num_batches = len(self._train_data_loader)

            batch: SynthSample
            for i, batch in enumerate(self._train_data_loader):

                # Turn on training mode
                self.model.train()

                # Batch number
                batch_number = i + 1

                # Zero the gradients
                optimizer.zero_grad()

                # Pass the batch through the model and
                # compute the loss
                loss = self._forward_batch(batch)

                # Backward pass
                loss.backward()

                # Clip gradients
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self._config.max_grad_norm,
                )

                # Update weights
                optimizer.step()

                # Add loss
                train_losses.append(loss.item())

                # Batch training log

                # Log record
                train_log_record = BatchTrainingLogRecord(
                    epoch=epoch,
                    batch_number=batch_number,
                    num_batches=num_batches,
                    train_loss=loss.item(),
                )

                # Loguru
                self._logger.info(train_log_record.to_message())

                # Wandb
                if self._config.wandb:
                    wandb.log(train_log_record.model_dump())

            # Epoch training log

            # Log record
            train_log_record = EpochTrainingLogRecord(
                epoch=epoch,
                lr=scheduler.get_last_lr()[0],
                avg_train_loss=sum(train_losses) / num_batches,
            )

            # Loguru
            self._logger.info(train_log_record.to_message())

            # Wandb
            if self._config.wandb:
                wandb.log(train_log_record.model_dump())

            # Validate
            self._validate(epoch)

            # Adjust learning rate
            scheduler.step()

            # Save checkpoint
            self._save_checkpoint(epoch)

    def _prepare_data(self) -> None:

        # Read all tokens
        with open(self._config.tokens_file_path, "r") as f:
            tokens = f.read().splitlines()

        # Text converter
        self._text_converter = TextConverter(tokens)

        # Load datasets
        train_dataset = SynthDataset(
            self._config.dataset_dir,
            self._config.train_samples_file_path,
            image_converter=self._image_converter,
            text_converter=self._text_converter,
        )
        valid_dataset = SynthDataset(
            self._config.dataset_dir,
            self._config.valid_samples_file_path,
            image_converter=self._image_converter,
            text_converter=self._text_converter,
        )

        # Create data loaders
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=self._config.train_batch_size,
            collate_fn=SynthDataset.collate,
        )
        valid_data_loader = DataLoader(
            valid_dataset,
            batch_size=self._config.valid_batch_size,
            collate_fn=SynthDataset.collate,
        )

        self._train_data_loader = train_data_loader
        self._valid_data_loader = valid_data_loader

        # Create model and
        # move to device
        self._model = CRNN(
            in_channels=self._config.in_channels,
            hidden_size=self._config.hidden_size,
            num_tokens=len(tokens),
            image_tensor_height=self._config.image_tensor_height,
            image_tensor_width=self._config.image_tensor_width,
        ).to(self.device)

    def _validate(self, epoch: int) -> None:

        # Turn on evaluation mode
        self.model.eval()

        with torch.no_grad():
            # List of loss of each batch
            valid_losses = []

            # Validation loop
            num_batches = len(self._valid_data_loader)
            batch: SynthSample
            for batch in self._valid_data_loader:

                # Pass the batch through the model and
                # compute the loss
                loss = self._forward_batch(batch)

                # Add to list
                valid_losses.append(loss.item())

            # Overall validation performance
            valid_loss = sum(valid_losses) / num_batches

            # Log

            # Log record
            valid_log_record = ValidationLogRecord(
                epoch=epoch,
                valid_loss=valid_loss,
            )

            # Loguru
            logger.info(valid_log_record.to_message())

            # Wandb
            if self._config.wandb:
                wandb.log(valid_log_record.model_dump())

    def _forward_batch(self, batch: SynthSample) -> Tensor:
        """Pass the batch through the model and compute the loss.

        Parameters
        ----------
        batch : SynthSample
            A batch of samples. Each field is a data batch.

        Returns
        -------
        Tensor
            CTC loss.
        """

        # Unpack batched samples, and
        # move data to device
        images = batch.image.to(self.device)
        encoded_texts = batch.encoded_text.to(self.device)
        text_lengths = batch.text_length.to(self.device)

        # Forward pass
        # Shape: (seq_len, batch_size, num_tokens)
        output: Tensor = self.model(images)

        # Get sequence length and batch size
        seq_len, batch_size, _ = output.shape

        # Compute the log-probabilities
        log_probs = F.log_softmax(output, dim=-1)

        # Compute loss
        loss: Tensor = F.ctc_loss(
            log_probs=log_probs,
            targets=encoded_texts,
            input_lengths=torch.full((batch_size,), seq_len).to(self.device),
            target_lengths=text_lengths,
            # * Zero the infinity loss and the associated gradients!
            zero_infinity=True,
        )

        return loss

    def _save_checkpoint(self, epoch: int) -> None:

        # No need to save
        if (
            epoch % self._config.save_every_n_epochs != 0
            and epoch != self._config.epochs
        ):
            return

        # State dict
        state_dict = self.model.state_dict()

        # File path
        checkpoint_file_path = self._config.checkpoints_dir.joinpath(
            f"checkpoint-{epoch}"
        ).with_suffix(".pt")

        # Save
        torch.save(state_dict, checkpoint_file_path)

        # Log
        self._logger.info(f"Checkpoint is saved at: {checkpoint_file_path}")

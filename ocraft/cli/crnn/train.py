from typing import Annotated
from pathlib import Path
import typer

from .app import crnn_app


@crnn_app.command(help="Train the CRNN model.")
def train(
    file: Annotated[
        Path,
        typer.Argument(
            show_default=False,
            help="Training configuration file (in the format of TOML).",
        ),
    ]
):
    from ...crnn.train import Trainer

    # Create a trainer
    trainer = Trainer.from_config_file(file)

    # Train!
    trainer.train()

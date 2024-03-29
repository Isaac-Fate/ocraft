from typing import Annotated, Optional
from pathlib import Path
import typer

from .app import crnn_app


@crnn_app.command(
    help=(
        "Split the Synth90K dataset into training and validation subsets. "
        "Two files, 'train-samples.csv' and 'valid-samples.csv', "
        "will be created in the run directory."
    ),
)
def split(
    dataset_dir: Annotated[
        Path,
        typer.Argument(
            show_default=False,
            help="Root directory of the Synth90K dataset.",
        ),
    ],
    run_dir: Annotated[
        Path,
        typer.Argument(
            show_default=False,
            help="Directory to save the split data subsets.",
        ),
    ],
    num_train_samples: Annotated[
        int,
        typer.Option(
            "--train",
            help="Number of training samples.",
        ),
    ] = 10000,
    num_valid_samples: Annotated[
        int,
        typer.Option(
            "--valid",
            help="Number of validation samples.",
        ),
    ] = 1000,
    random_seed: Annotated[
        Optional[int],
        typer.Option(
            "--seed",
            help="Random seed.",
        ),
    ] = None,
) -> None:

    # Create run directory if it does not exist
    run_dir.mkdir(parents=True, exist_ok=True)

    # Annotation file path
    annotation_file_path = dataset_dir.joinpath("annotation.csv")

    # Load all sample annotations
    with open(annotation_file_path, "r") as f:
        rows = f.read().splitlines()

    # Skip header
    rows = rows[1:]

    # Import NumPy
    import numpy as np

    # Random generator
    rng = np.random.default_rng(random_seed)

    # Select training and validation samples
    sample_indices = rng.choice(
        len(rows),
        size=num_train_samples + num_valid_samples,
        replace=False,
    )

    # Select training samples
    train_sample_indices = sample_indices[:num_train_samples]

    # Select validation samples
    valid_sample_indices = sample_indices[num_train_samples:]

    # Write to files

    from ...crnn.data import SynthSampleMeta

    # Get CSV header
    header = SynthSampleMeta._fields

    with open(run_dir.joinpath("train-samples.csv"), "w") as f:
        # Write header
        f.write(",".join(header) + "\n")

        # Write rows
        for i in train_sample_indices:
            row = rows[i]
            f.write(f"{row}\n")

    with open(run_dir.joinpath("valid-samples.csv"), "w") as f:
        # Write header
        f.write(",".join(header) + "\n")

        # Write rows
        for i in valid_sample_indices:
            row = rows[i]
            f.write(f"{row}\n")

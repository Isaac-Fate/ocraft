from typing import Annotated, Optional
from pathlib import Path
import re
import typer
import rich
from rich.progress import track

from .app import crnn_app


def is_chinese_char(char: str) -> bool:
    return re.match(r"^[\u4e00-\u9fff]$", char) is not None


@crnn_app.command(
    help="Synthesize images with given tokens.",
)
def synth(
    tokens_file_path: Annotated[
        Path,
        typer.Argument(
            show_default=False,
            help="Path of the file consisting of the tokens.",
        ),
    ],
    fonts_dir: Annotated[
        Path,
        typer.Argument(
            show_default=False,
            help="Directory of all the TTF fonts in the system.",
        ),
    ],
    num_samples: Annotated[
        int,
        typer.Option(
            "-n",
            "--num",
            help="Number of samples to generate.",
        ),
    ] = 1000,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--out",
            help="Output directory. If not set, it will be saved in ./synth-dataset under the current directory.",
        ),
    ] = None,
    min_text_len: Annotated[
        int,
        typer.Option(
            "-l",
            "--len-min",
            help="Minimum length of the text.",
        ),
    ] = 3,
    max_text_len: Annotated[
        int,
        typer.Option(
            "-L",
            "--len-max",
            help="Maximum length of the text.",
        ),
    ] = 15,
    font_size: Annotated[
        int,
        typer.Option(
            "-s",
            "--font-size",
            help="Font size",
        ),
    ] = 10,
    grayscale_value: Annotated[
        float,
        typer.Option(
            "-g",
            "--gray",
            help="Grayscale value of the image to synthesize. This will be ignored if --gray-min and --gray-max are set.",
        ),
    ] = 1.0,
    min_grayscale_value: Annotated[
        Optional[float],
        typer.Option(
            "-g",
            "--gray-min",
            help="Lower bound of the grayscale value of the image to synthesize.",
        ),
    ] = None,
    max_grayscale_value: Annotated[
        Optional[float],
        typer.Option(
            "-G",
            "--gray-max",
            help="Upper bound of the grayscale value of the image to synthesize.",
        ),
    ] = None,
    sum_chinese_token_probs: Annotated[
        float,
        typer.Option(
            "--zh-prob",
            help="Sum of probabilities associated with Chinese characters.",
        ),
    ] = 0.1,
    random_seed: Annotated[
        Optional[int],
        typer.Option(
            "--seed",
            help="Random seed.",
        ),
    ] = None,
):
    # The output directory is set as the current directory if not specified
    if output_dir is None:
        output_dir = Path.cwd().joinpath("synth-dataset")

        # Create output directory
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except:
            rich.print(f"[red]Output directory {output_dir} already exists. Exiting...")
            exit(0)

    # Load all tokens
    with open(tokens_file_path, "r") as f:
        tokens = f.read().splitlines()

    # Number of Chinese tokens/characters
    num_chinese_tokens = sum(is_chinese_char(char) for char in tokens)

    # Number of other tokens
    num_other_tokens = len(tokens) - num_chinese_tokens

    # Sum of probabilities associated with symbols, digits and letters
    sum_other_token_probs = 1 - sum_chinese_token_probs

    # Import NumPy
    import numpy as np

    # Probabilities associated with Chinese characters
    chinese_token_probs = np.full(
        num_chinese_tokens, sum_chinese_token_probs / num_chinese_tokens
    )

    # Probabilities associated with symbols, digits and letters
    other_token_probs = np.full(
        num_other_tokens, sum_other_token_probs / num_other_tokens
    )

    # Probabilities associated with all tokens
    probs = np.concatenate((other_token_probs, chinese_token_probs)).tolist()

    # Random generator
    rng = np.random.default_rng(random_seed)

    from ...crnn.data import SynthSampleMeta

    sample_metas: list[SynthSampleMeta] = []
    for index in track(range(num_samples), description="Generating texts..."):

        # Random text length
        text_len = rng.integers(min_text_len, max_text_len + 1)

        # Generate random text
        text = "".join(
            rng.choice(
                tokens,
                size=text_len,
                replace=True,
                p=probs,
            )
        )

        # Name image file path
        image_file_path = f"./images/{index}.jpg"

        # Sample meta
        sample_meta = SynthSampleMeta(
            image_file_path=image_file_path,
            text=text,
        )

        sample_metas.append(sample_meta)

    # Write annotation file
    with open(output_dir.joinpath("annotation.csv"), "w") as f:

        # Write header
        header = ",".join(SynthSampleMeta._fields)
        f.write(f"{header}\n")

        # Write samples
        for sample_meta in track(sample_metas, description="Writing annotation..."):
            f.write(f"{sample_meta.image_file_path},{sample_meta.text}\n")

    # Import the synthesizer
    from ...crnn.data import ImageSynthesizer

    # Create synthesizer
    image_synthesizer = ImageSynthesizer(
        # Use all the TTF fonts under the fonts directory
        list(fonts_dir.glob("*.ttf")),
        random_seed=random_seed,
    )

    # Images directory
    images_dir = output_dir.joinpath("images")
    images_dir.mkdir(parents=True, exist_ok=True)

    # Synthesize images
    for sample_meta in track(sample_metas, description="Synthesizing images..."):

        # Generate random grayscale value if required
        if min_grayscale_value is not None and max_grayscale_value is not None:
            grayscale_value = rng.uniform(min_grayscale_value, max_grayscale_value)

        # Synthesize image
        image = image_synthesizer.synthesize(
            sample_meta.text,
            font_size=font_size,
            grayscale_value=grayscale_value,
        )

        # Save image
        image.save(output_dir.joinpath(sample_meta.image_file_path))

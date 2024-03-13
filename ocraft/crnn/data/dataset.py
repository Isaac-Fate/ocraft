from collections import namedtuple
from pathlib import Path
from PIL import Image
import csv
import torch
from torch.utils.data import Dataset

from .image_converter import ImageConverter
from .text_converter import TextConverter


Synth90kSampleMeta = namedtuple(
    "Synth90kSampleMeta",
    (
        "image_file_path",
        "text",
    ),
)


Synth90kSample = namedtuple(
    "Synth90kSample",
    (
        "image",
        "encoded_text",
        "text_length",
    ),
)


Synth90kRawSample = namedtuple(
    "Synth90kRawSample",
    (
        "image",
        "text",
    ),
)


class Synth90kDataset(Dataset):

    def __init__(
        self,
        dataset_dir: Path,
        annotation_file_path: Path,
        *,
        image_converter: ImageConverter,
        text_converter: TextConverter,
    ) -> None:

        super().__init__()

        self._dataset_dir = dataset_dir
        self._annotation_file_path = annotation_file_path
        self._image_converter = image_converter
        self._text_converter = text_converter

        # Load annotations
        self._sample_metas: list[Synth90kSampleMeta] = []
        with open(self._annotation_file_path, "r") as f:

            # Create a CSV reader
            reader = csv.reader(f)

            # Skip header
            next(reader)

            for row in reader:
                # The image file path is relative to the synth90k dataset directory
                # We do not load the absolute path here to increase performance
                sample_meta = Synth90kSampleMeta(*row)

                # Add sample annotation
                self._sample_metas.append(sample_meta)

    def __len__(self) -> int:

        return len(self._sample_metas)

    def __getitem__(self, index: int) -> Synth90kSample:

        # Get sample annotation
        sample_meta = self._sample_metas[index]

        # Load image
        image_file_path = self._dataset_dir.joinpath(sample_meta.image_file_path)
        image = Image.open(image_file_path)

        # Transform image to tensor
        image = self._image_converter.transform(image)

        # Encode text to tensor
        encoded_text = self._text_converter.encode(sample_meta.text)

        # Calculate text length, and then
        # convert to tensor
        text_length = len(encoded_text)
        text_length = torch.LongTensor([text_length])

        return Synth90kSample(
            image=image,
            encoded_text=encoded_text,
            text_length=text_length,
        )

    @staticmethod
    def collate(samples: list[Synth90kSample]) -> Synth90kSample:
        """Convert list of samples to a batch.

        Parameters
        ----------
        samples : list[Synth90kSample]
            A list of samples.

        Returns
        -------
        Synth90kSample
            A batch of samples.
            - image: A tensor of shape (N, 1, H, W).
            - encoded_text: A tensor of shape (L,) where L is the sum of all text lengths.
            - text_length: A tensor of shape (N,).
        """

        # Convert list of samples to a batch, i.e,
        # convert row-oriented data to column-oriented data
        batch = Synth90kSample(*(zip(*samples)))

        # Unpack batch
        images = batch.image
        encoded_texts = batch.encoded_text
        text_lengths = batch.text_length

        # Stack image tensors
        # Shape: (N, 1, H, W)
        images = torch.stack(images, dim=0)

        # Concatenate encoded text tensors
        # Shape: (L,) where L is the sum of all text lengths
        encoded_texts = torch.cat(encoded_texts, dim=0)

        # Concatenate text length tensors
        # Shape: (N,)
        text_lengths = torch.cat(text_lengths, dim=0)

        # Form a new batch
        batch = Synth90kSample(
            image=images,
            encoded_text=encoded_texts,
            text_length=text_lengths,
        )

        return batch

    def get_raw_sample(self, index: int) -> Synth90kRawSample:
        """Get the unprocessed raw sample.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        Synth90kRawSample
            - image: A PIL Image object.
            - text: The text string in the image.
        """

        # Get sample annotation
        sample_meta = self._sample_metas[index]

        # Load image
        image_file_path = self._dataset_dir.joinpath(sample_meta.image_file_path)
        image = Image.open(image_file_path)

        return Synth90kRawSample(
            image=image,
            text=sample_meta.text,
        )

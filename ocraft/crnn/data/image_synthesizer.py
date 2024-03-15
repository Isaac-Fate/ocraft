from typing import Optional, Iterable
from collections import namedtuple
from pathlib import Path
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
import numpy as np


BBox = namedtuple(
    "BBox",
    (
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ),
)


class ImageSynthesizer:

    def __init__(
        self,
        font_file_paths: list[Path],
        *,
        random_seed: Optional[int] = None,
    ) -> None:

        self._font_file_paths = font_file_paths

        self._fonts = []
        for font_file_path in font_file_paths:
            font = TTFont(font_file_path)
            self._fonts.append(font)

        # Random generator
        self._rng = np.random.RandomState(random_seed)

    @staticmethod
    def has_glyph(
        font: TTFont,
        glyph: str,
    ) -> bool:

        # Interate over all cmaps (Character to Glyph Index Mapping Tables) in the font
        for table in font["cmap"].tables:

            # Return True if the unicode of the glyph is found in the cmap table
            if ord(glyph) in table.cmap.keys():
                return True

        return False

    def synthesize(
        self,
        text: str,
        *,
        font_size: int = 10,
        grayscale_value: float | tuple[float, float] = 1.0,
    ):

        # Create a dummy image so that
        # we can create an `ImageDraw` object
        dummy_image = Image.new("L", (0, 0), 0)

        # Create an `ImageDraw` object
        image_draw = ImageDraw.Draw(dummy_image)

        # Randomly permute the font indices
        font_indices = self._rng.permutation(len(self._fonts))

        # Determine the font of each token
        token_image_fonts = []
        for token in text:
            font_index = self._select_font_index(token, font_indices)
            font_file_path = self._font_file_paths[font_index]
            image_font = ImageFont.truetype(
                font_file_path,
                size=font_size,
            )
            token_image_fonts.append(image_font)

        # Bounding boxes of all tokens in the text
        bboxes: list[BBox] = []
        x, y = (0, 0)
        for i, token in enumerate(text):
            # Get the image font of the token
            image_font = token_image_fonts[i]

            # Get the bounding box of the token
            bbox = BBox(*image_draw.textbbox((x, y), token, font=image_font))

            # Add the bounding box
            bboxes.append(bbox)

            # Update x
            x = bbox.x_max

        # Compute the bounding box for the whole image
        bboxes_arr = np.array(bboxes)
        image_bbox = BBox(*bboxes_arr.min(axis=0)[:2], *bboxes_arr.max(axis=0)[2:])

        # Randomly generate a grayscale value if the it is a tuple of low and high values
        if isinstance(grayscale_value, tuple):
            low, high = grayscale_value
            grayscale_value = self._rng.uniform(low, high)

        # Create the image to draw on
        image = Image.new(
            mode="L",
            size=(image_bbox.x_max, image_bbox.y_max),
            # Conver to pixel value
            color=int(grayscale_value * 255),
        )

        # Create an `ImageDraw` object
        image_draw = ImageDraw.Draw(image)

        # Reset x and y
        x, y = (0, 0)
        for i, token in enumerate(text):
            # Get the image font of the token
            image_font = token_image_fonts[i]

            # Draw the token on the image
            image_draw.text((x, y), token, font=image_font)

            # Get the bounding box for this token
            bbox = bboxes[i]

            # Update x
            x = bbox.x_max

        return image

    def _select_font_index(
        self,
        token: str,
        font_indices: Iterable[int],
    ) -> int:

        for index in font_indices:

            # Get the font
            font = self._fonts[index]

            # Return this index since the associated font supports the token
            if self.has_glyph(font, token):
                return index

        raise ValueError(f"Token {token} is not found in any of the provided fonts")

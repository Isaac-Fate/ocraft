from PIL import Image
from torch import Tensor
from torchvision import transforms


class ImageConverter:

    def __init__(
        self,
        image_tensor_height: int,
        image_tensor_width: int,
    ) -> None:

        self._image_tensor_height = image_tensor_height
        self.image_tensor_width = image_tensor_width

        self._converter = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_tensor_height, image_tensor_width)),
            ]
        )

    def __call__(self, image: Image) -> Tensor:

        return self.transform(image)

    def transform(self, image: Image.Image) -> Tensor:
        """Transform image to tensor.

        Parameters
        ----------
        image : Image.Image
            Input image.

        Returns
        -------
        Tensor
            Shape: (1, H, W) or (N, H, W)
        """

        # Convert image to grayscale
        image = image.convert("L")

        # Convert image to tensor
        image_tensor = self._converter(image)

        return image_tensor

    def inverse_transform(self, image_tensor: Tensor) -> Image:

        pass

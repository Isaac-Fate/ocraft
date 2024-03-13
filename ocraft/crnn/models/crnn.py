from collections import OrderedDict
from torch import Tensor
from torch import nn

from .bidirectional_lstm import BidirectionalLSTM


class CRNN(nn.Module):

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_size: int,
        num_tokens: int,
        image_tensor_height: int,
        image_tensor_width: int,
    ):

        super().__init__()

        self._num_tokens = num_tokens
        self._image_tensor_height = image_tensor_height
        self._image_tensor_width = image_tensor_width

        # Add CNN blocks
        self.cnn = nn.Sequential(
            OrderedDict(
                # -------
                # Block 1
                # -------
                conv1=nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=64,
                    kernel_size=3,
                    padding=1,
                ),
                relu1=nn.ReLU(inplace=True),
                pool1=nn.MaxPool2d(kernel_size=2, stride=2),
                # Ouput shape: (N, 64, H/2, W/2)
                #
                # -------
                # Block 2
                # -------
                conv2=nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=3,
                    padding=1,
                ),
                relu2=nn.ReLU(inplace=True),
                pool2=nn.MaxPool2d(kernel_size=2, stride=2),
                # Ouput shape: (N, 128, H/4, W/4)
                #
                # -------
                # Block 3
                # -------
                conv3=nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    padding=1,
                ),
                bn3=nn.BatchNorm2d(256),
                relu3=nn.ReLU(inplace=True),
                # Ouput shape: (N, 256, H/4, W/4)
                #
                # -------
                # Block 4
                # -------
                conv4=nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    padding=1,
                ),
                relu4=nn.ReLU(inplace=True),
                pool4=nn.MaxPool2d(
                    kernel_size=2,
                    stride=(2, 1),
                    padding=(0, 1),
                ),
                # Ouput shape: (N, 256, H/8, _)
                #
                # ------
                # Block 5
                # ------
                conv5=nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=3,
                    padding=1,
                ),
                bn5=nn.BatchNorm2d(512),
                relu5=nn.ReLU(inplace=True),
                # Ouput shape: (N, 512, H/8, _)
                #
                # -------
                # Block 6
                # -------
                conv6=nn.Conv2d(
                    in_channels=512,
                    out_channels=512,
                    kernel_size=3,
                    padding=1,
                ),
                relu6=nn.ReLU(inplace=True),
                pool6=nn.MaxPool2d(
                    kernel_size=2,
                    stride=(2, 1),
                    padding=(0, 1),
                ),
                # Ouput shape: (N, 512, H/16, _)
                #
                # ------
                # Block 7
                # ------
                conv7=nn.Conv2d(
                    in_channels=512,
                    out_channels=512,
                    kernel_size=2,
                    stride=1,
                    padding=0,
                ),
                bn7=nn.BatchNorm2d(512),
                relu7=nn.ReLU(inplace=True),
                # Ouput shape: (N, 512, H/16, _)
            )
        )

        self.rnn = nn.Sequential(
            OrderedDict(
                bidirectional_lstm1=BidirectionalLSTM(
                    input_size=512,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                ),
                bidirectional_lstm2=BidirectionalLSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    # * +1 for the blank token
                    output_size=num_tokens + 1,
                ),
            )
        )

    @property
    def num_tokens(self) -> int:
        """Total number of tokens excluding the blank token."""

        return self._num_tokens

    @property
    def image_tensor_height(self) -> int:
        """Height of the input image tensor."""

        return self._image_tensor_height

    @property
    def image_tensor_width(self) -> int:
        """Width of the input image tensor."""

        return self._image_tensor_width

    def forward(self, image: Tensor) -> Tensor:

        # Get output from CNN
        cnn_output: Tensor = self.cnn(image)

        # Check output shape
        batch_size, num_channels, height, width = cnn_output.shape
        assert height == 1, "The height of the CNN output must be 1"

        # (batch_size, num_channels, width)
        sequence = cnn_output.squeeze(2)

        # (width, batch_size, num_channels), i.e.,
        # (seq_len, batch_size, input_size)
        sequence = sequence.permute(2, 0, 1)

        # Get output from RNN
        # (width, batch_size, num_tokens), i.e.,
        # (seq_len, batch_size, output_size)
        output = self.rnn(sequence)

        return output

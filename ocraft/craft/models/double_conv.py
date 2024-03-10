from torch import Tensor
from torch import nn


class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels_part1: int,
        in_channels_part2: int,
        out_channels: int,
    ):

        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            # First conv
            nn.Conv2d(
                in_channels_part1 + in_channels_part2,
                in_channels_part2,
                kernel_size=1,
            ),
            nn.BatchNorm2d(in_channels_part2),
            nn.ReLU(inplace=True),
            # Ouput shape: (N, in_channels_part2, H, W)
            #
            # Second conv
            nn.Conv2d(
                in_channels_part2,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Ouput shape: (N, out_channels, H, W)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation.

        Parameters
        ----------
        x : Tensor
            Shape: (N, in_channels_part1 + in_channels_part2, H, W)

        Returns
        -------
        Tensor
            Shape: (N, out_channels, H, W)
        """

        return self.conv(x)

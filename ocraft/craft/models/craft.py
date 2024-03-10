from collections import namedtuple
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from .vgg16_bn_backbone import VGG16BNBackbone, VGG16BNBackboneOutput
from .double_conv import DoubleConv


CRAFTOutput = namedtuple(
    "CRAFTOutput",
    (
        "feature",
        "region_score",
        "affinity_score",
    ),
)


class CRAFT(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        # VGG16 backbone
        self.backbone = VGG16BNBackbone()

        # Upsampling conv layers
        self.up_conv1 = DoubleConv(1024, 512, 256)
        self.up_conv2 = DoubleConv(512, 256, 128)
        self.up_conv3 = DoubleConv(256, 128, 64)
        self.up_conv4 = DoubleConv(128, 64, 32)

        # Top layer
        self.top = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
        )

    def forward(self, image: Tensor) -> CRAFTOutput:
        """
        Forward propagation.

        Parameters
        ----------
        image : Tensor
            Image tensor with shape (N, 3, H, W).

        Returns
        -------
        CRAFTOutput
            - feature: (N, 1024+512, H/16, W/16)
            - region_score: (N, H/16, W/16), the probability that the given pixel is the center of the character.
            - affinity_score: (N, H/16, W/16), the center probability of the space between adjacent characters.
        """

        # Get backbone output
        # - relu2_2: (N, 128, H/2, W/2)
        # - relu3_2: (N, 256, H/4, W/4)
        # - relu4_2: (N, 512, H/8, W/8)
        # - relu5_2: (N, 512, H/16, W/16)
        # - out: (N, 1024, H/16, W/16)
        backbone_output: VGG16BNBackboneOutput = self.backbone(image)

        # (N, 1024+512, H/16, W/16)
        x = torch.concat((backbone_output.out, backbone_output.relu5_2), dim=1)

        # (N, 256, H/16, W/16)
        x = self.up_conv1(x)

        # (N, 256, H/8, W/8)
        x = F.interpolate(
            x,
            size=backbone_output.relu4_2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # (N, 256+512, H/8, W/8)
        x = torch.concat((x, backbone_output.relu4_2), dim=1)

        # (N, 128, H/8, W/8)
        x = self.up_conv2(x)

        # (N, 128, H/4, W/4)
        x = F.interpolate(
            x,
            size=backbone_output.relu3_2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # (N, 128+256, H/4, W/4)
        x = torch.concat((x, backbone_output.relu3_2), dim=1)

        # (N, 64, H/4, W/4)
        x = self.up_conv3(x)

        # (N, 64, H/2, W/2)
        x = F.interpolate(
            x,
            size=backbone_output.relu2_2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # (N, 64+128, H/2, W/2)
        x = torch.concat((x, backbone_output.relu2_2), dim=1)

        # (N, 32, H/2, W/2)
        feature = self.up_conv4(x)

        # (N, 2, H/2, W/2)
        out = self.top(feature)

        # Split the out into region and affinity scores
        # - region_score: (N, H/2, W/2)
        # - affinity_score: (N, H/2, W/2)
        region_score = out[:, 0, :, :]
        affinity_score = out[:, 1, :, :]

        return CRAFTOutput(
            feature=feature,
            region_score=region_score,
            affinity_score=affinity_score,
        )

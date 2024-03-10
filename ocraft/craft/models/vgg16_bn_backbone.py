from enum import IntEnum
from collections import namedtuple
from torch import Tensor
from torch import nn
from torchvision.models import vgg16_bn


VGG16BNBackboneOutput = namedtuple(
    "VGG16BNFeatureOutput",
    (
        "relu2_2",
        "relu3_2",
        "relu4_2",
        "relu5_2",
        "out",
    ),
)


class VGG16FeatureLayerName(IntEnum):

    conv1_1 = 0
    bn1_1 = 1
    relu1_1 = 2
    conv1_2 = 3
    bn1_2 = 4
    relu1_2 = 5
    pool1 = 6
    conv2_1 = 7
    bn2_1 = 8
    relu2_1 = 9
    conv2_2 = 10
    bn2_2 = 11
    relu2_2 = 12
    pool2 = 13
    conv3_1 = 14
    bn3_1 = 15
    relu3_1 = 16
    conv3_2 = 17
    bn3_2 = 18
    relu3_2 = 19
    conv3_3 = 20
    bn3_3 = 21
    relu3_3 = 22
    pool3 = 23
    conv4_1 = 24
    bn4_1 = 25
    relu4_1 = 26
    conv4_2 = 27
    bn4_2 = 28
    relu4_2 = 29
    conv4_3 = 30
    bn4_3 = 31
    relu4_3 = 32
    pool4 = 33
    conv5_1 = 34
    bn5_1 = 35
    relu5_1 = 36
    conv5_2 = 37
    bn5_2 = 38
    relu5_2 = 39
    conv5_3 = 40
    bn5_3 = 41
    relu5_3 = 42
    pool5 = 43


class VGG16BNBackbone(nn.Module):

    def __init__(self):

        super().__init__()

        # Whole model
        model = vgg16_bn()

        # We only need feature layers
        features = model.features

        self.block1 = features[
            self.slice(VGG16FeatureLayerName.conv1_1, VGG16FeatureLayerName.relu2_2)
        ]

        self.block2 = features[
            self.slice(VGG16FeatureLayerName.pool2, VGG16FeatureLayerName.relu3_2)
        ]

        self.block3 = features[
            self.slice(VGG16FeatureLayerName.conv3_3, VGG16FeatureLayerName.relu4_2)
        ]

        self.block4 = features[
            self.slice(VGG16FeatureLayerName.conv4_3, VGG16FeatureLayerName.relu5_2)
        ]

        self.block5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )

    def forward(self, image: Tensor) -> VGG16BNBackboneOutput:
        """
        Forward propagation.

        Parameters
        ----------
        image : Tensor
            Shape: (N, 3, H, W)

        Returns
        -------
        VGG16BNBackboneOutput
            - relu2_2: (N, 128, H/2, W/2)
            - relu3_2: (N, 256, H/4, W/4)
            - relu4_2: (N, 512, H/8, W/8)
            - relu5_2: (N, 512, H/16, W/16)
            - out: (N, 1024, H/16, W/16)
        """

        # (N, 128, H/2, W/2)
        relu2_2 = self.block1(image)

        # (N, 256, H/4, W/4)
        relu3_2 = self.block2(relu2_2)

        # (N, 512, H/8, W/8)
        relu4_2 = self.block3(relu3_2)

        # (N, 512, H/16, W/16)
        relu5_2 = self.block4(relu4_2)

        # (N, 1024, H/16, W/16)
        out = self.block5(relu5_2)

        return VGG16BNBackboneOutput(
            relu2_2=relu2_2,
            relu3_2=relu3_2,
            relu4_2=relu4_2,
            relu5_2=relu5_2,
            out=out,
        )

    @staticmethod
    def slice(
        start_layer: VGG16FeatureLayerName,
        end_layer: VGG16FeatureLayerName,
    ) -> slice:
        """Get a slice of layers. The end layer is included!

        Parameters
        ----------
        start_layer : VGG16FeatureLayerName
            Starting layer.
        end_layer : VGG16FeatureLayerName
            End layer.

        Returns
        -------
        slice
            Slice of layers in between start and end layer.
            The start and end layer are also included.
        """

        return slice(start_layer, end_layer + 1)

    @staticmethod
    def init_weights(modules: list[nn.Module]) -> None:
        """
        Initialize weights for model.

        Parameters
        ----------
        modules : list[nn.Module]
            List of modules to initialize.
        """

        for m in modules:
            # Initialize weights for conv using Xavier distribution
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

            # Initialize weights for batch norm by setting its weight to 1 and bias to 0
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            # Initialize weights for linear layer using normal distribution
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

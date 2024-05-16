from torch import nn


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        residual=True,
        act_layer=nn.ReLU,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        if not residual:
            self.res = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.res = lambda x: x
        else:
            self.res = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act = act_layer()

    def forward(self, x):
        res = self.res(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x)) + res

        return x


class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Conv2d(in_channels, 64, 7, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, 2, 1),
                ResBlock(64, 64, 3, 2),
                # nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
                ResBlock(64, 64, 3, 1),
                ResBlock(64, 128, 3, 2),
                ResBlock(128, 128, 3, 1),
                # nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
                ResBlock(128, 256, 3, 2),
                # nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
                ResBlock(256, 256, 3, 1),
                ResBlock(256, 512, 3, 2),
                ResBlock(512, 512, 3, 1),
            ]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(
            512,
            num_classes,
        )
        self.act = nn.ReLU()

    def forward(self, x):
        for blk in self.blocks:
            x = self.act(blk(x))
        x = self.avg_pool(x).squeeze(dim=(2, 3))

        x = self.head(x)
        return x

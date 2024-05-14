from torch import nn


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        residual=True,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        if not residual:
            self.res = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.res = lambda x: x
        else:
            self.res = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        res = self.res(x)

        x = self.norm(self.conv(x)) + res

        return x


class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResBlock(in_channels, 64, 3, 1),
                nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
                ResBlock(64, 64, 3, 1),
                ResBlock(64, 128, 3, 1),
                nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
                ResBlock(128, 128, 3, 1),
                ResBlock(128, 256, 3, 2),
                nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
                ResBlock(256, 256, 3, 1),
            ]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(
            256,
            num_classes,
        )
        self.act = nn.ReLU()

    def forward(self, x):
        for blk in self.blocks:
            x = self.act(blk(x))
        x = self.avg_pool(x).squeeze(dim=(2, 3))

        x = self.head(x)
        return x

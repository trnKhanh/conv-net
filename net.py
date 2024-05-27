from torch import nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        depth: int = 1,
        groups: int = 1,
        residual=True,
        dropout_rate: float = 0,
    ):
        super().__init__()
        self.act = nn.ReLU()
        self.drop = (
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        padding = kernel_size // 2

        # If the input and output shapes are different, then 1x1 conv must be
        # used to transform the input for skip connection
        if not residual:
            self.res = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.res = lambda x: x
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels),
            )

        self.convs = nn.ModuleList()
        # First layer is used for input transformation
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size if groups == 1 else 1,
                    stride,
                    padding if groups == 1 else 0,
                    groups=1,
                ),
                nn.BatchNorm2d(out_channels),
            )
        )
        # Add layers based on depth
        for _ in range(depth):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size,
                        1,
                        padding,
                        groups=groups,
                    ),
                    nn.BatchNorm2d(out_channels),
                )
            )

        if groups > 1:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        1,
                        1,
                    ),
                    nn.BatchNorm2d(out_channels),
                )
            )

    def forward(self, x):
        res = self.res(x)

        for conv in self.convs:
            x = self.drop(self.act(conv(x)))

        return x + res


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        res_depth: int = 1,
        groups: int = 1,
        depth: int = 1,
        residual=True,
        dropout_rate: float = 0,
        first_block: bool = False,
    ):
        super().__init__()
        self.act = nn.ReLU()
        self.drop = (
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

        self.res_blocks = nn.ModuleList()
        # Add <depth> ResidualBlock
        # If this is first block, then we do not use skip connection
        for i in range(depth):
            self.res_blocks.append(
                ResidualBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    res_depth,
                    groups,
                    False if i == 0 and first_block else residual,
                    dropout_rate,
                )
            )

    def forward(self, x):
        for blk in self.res_blocks:
            x = self.drop(self.act(blk(x)))

        return x


class Net(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        net_configs,
        mlp_configs,
        mlp_dropout_rate: float = 0,
        conv_dropout_rate: float = 0,
        max_pool_stride: int = 1,
        init_down=None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.drop = (
            nn.Dropout(mlp_dropout_rate)
            if mlp_dropout_rate > 0
            else nn.Identity()
        )
        # The following lines of code parse the configuration and
        # create the corresponding architecture
        last_channels = in_channels
        if init_down is not None:
            for i, cf in enumerate(init_down):
                down_type = cf[0]
                if down_type == "conv":
                    conv_in_channels = last_channels
                    conv_out_channels = cf[1]
                    kernel_size = cf[2]
                    stride = cf[3]
                    self.blocks.append(
                        nn.Sequential(
                            nn.Conv2d(
                                conv_in_channels,
                                conv_out_channels,
                                kernel_size,
                                stride,
                                padding=kernel_size // 2,
                            ),
                            nn.BatchNorm2d(conv_out_channels),
                        )
                    )
                    last_channels = conv_out_channels

                elif down_type == "max":
                    kernel_size = cf[1]
                    stride = cf[2]
                    self.blocks.append(
                        nn.MaxPool2d(kernel_size, stride, kernel_size // 2)
                    )

        for i, cf in enumerate(net_configs):
            block_in_channels = last_channels
            block_out_channels = cf[0]
            kernel_size = cf[1]
            stride = cf[2]
            res_depth = cf[3]
            groups = cf[4]
            depth = cf[5]
            self.blocks.append(
                Block(
                    block_in_channels,
                    block_out_channels,
                    kernel_size,
                    stride,
                    res_depth,
                    groups,
                    depth,
                    True,
                    conv_dropout_rate,
                    i == 0,
                )
            )
            if max_pool_stride > 1 and i + 1 != len(net_configs):
                self.blocks.append(
                    nn.MaxPool2d(
                        kernel_size, max_pool_stride, kernel_size // 2
                    ),
                )
            last_channels = block_out_channels

        self.post_conv = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.ModuleList()
        self.last_channels = last_channels

        for i, channels in enumerate(mlp_configs):
            linear_in_channels = last_channels
            linear_out_channels = channels

            self.mlp.append(
                nn.Sequential(
                    nn.Linear(linear_in_channels, linear_out_channels),
                    nn.BatchNorm1d(linear_out_channels),
                )
            )
            last_channels = linear_out_channels

        self.head = nn.Linear(
            last_channels,
            num_classes,
        )

        self.act = nn.ReLU()

    def create_head(self, num_classes: int, mlp_configs):
        last_channels = self.last_channels
        self.mlp = nn.ModuleList()
        for i, channels in enumerate(mlp_configs):
            linear_in_channels = last_channels
            linear_out_channels = channels

            self.mlp.append(
                nn.Sequential(
                    nn.Linear(linear_in_channels, linear_out_channels),
                    nn.BatchNorm1d(linear_out_channels),
                )
            )
            last_channels = linear_out_channels

        self.head = nn.Linear(
            last_channels,
            num_classes,
        )

    def forward(self, x):
        for blk in self.blocks:
            x = self.act(blk(x))

        x = self.post_conv(x).squeeze(dim=(2, 3))

        for i, linear in enumerate(self.mlp):
            x = self.drop(self.act(linear(x)))

        x = self.head(x)

        return x

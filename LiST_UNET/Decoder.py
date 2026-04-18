import torch.nn as nn

from LiST_UNET_Block import Block


class TransposedConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r):
        super(TransposedConvLayer, self).__init__()
        self.transposed = nn.ConvTranspose3d(
            dim_in, dim_out, kernel_size=r, stride=r
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)

    def forward(self, x):
        x = self.transposed(x)
        x = self.norm(x)
        return x


class RecDecoder(nn.Module):
    def __init__(
        self,
        rec_channels=3,
        embed_dim=384,
        channels=(48, 96, 240),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 8),
        r=(4, 2, 2, 1),
        dropout=0.3,
    ):
        super(RecDecoder, self).__init__()

        # Reconstruction Head
        self.RecHead = TransposedConvLayer(
            dim_in=channels[0], dim_out=rec_channels, r=2
        )

        # Upsampling layers
        self.upconv3 = TransposedConvLayer(
            dim_in=channels[1], dim_out=channels[0], r=2
        )
        self.upconv2 = TransposedConvLayer(
            dim_in=channels[2], dim_out=channels[1], r=2
        )
        self.upconv1 = TransposedConvLayer(
            dim_in=embed_dim, dim_out=channels[2], r=2
        )

        # Transformer / UNet Blocks
        block = []
        for _ in range(blocks[0]):
            block.append(Block(channels=channels[0], r=r[0], heads=heads[0]))
        self.block1 = nn.Sequential(*block)

        block = []
        for _ in range(blocks[1]):
            block.append(Block(channels=channels[1], r=r[1], heads=heads[1]))
        self.block2 = nn.Sequential(*block)

        block = []
        for _ in range(blocks[2]):
            block.append(Block(channels=channels[2], r=r[2], heads=heads[2]))
        self.block3 = nn.Sequential(*block)

        block = []
        for _ in range(blocks[3]):
            block.append(Block(channels=embed_dim, r=r[3], heads=heads[3]))
        self.block4 = nn.Sequential(*block)

    def forward(self, x, hidden_states_out, x_shape):
        B, C, W, H, Z = x_shape

        x = x.reshape(B, C, W, H, Z)

        # Deepest block
        x = self.block4(x)

        # Decoder pathway with skip connections
        x = self.upconv1(x)
        x = x + hidden_states_out[2]

        x = self.block3(x)
        x = self.upconv2(x)
        x = x + hidden_states_out[1]

        x = self.block2(x)
        x = self.upconv3(x)
        x = x + hidden_states_out[0]

        x = self.block1(x)

        # Reconstruction output
        rec_output = self.RecHead(x)

        return rec_output
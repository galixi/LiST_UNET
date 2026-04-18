import torch
import torch.nn as nn
import scipy.io as io
import numpy as np

from LiST_UNET_Block import Block


class DepthwiseConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r):
        super(DepthwiseConvLayer, self).__init__()
        self.depth_wise = nn.Conv3d(dim_in, dim_out, kernel_size=r, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.norm(x)
        return x


class RecEncoder(nn.Module):
    def __init__(
        self,
        input_channels=4,
        embed_dim=384,
        embedding_dim=27,
        channels=(48, 96, 240),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 8),
        r=(4, 2, 2, 1),
        dropout=0.3,
    ):
        super(RecEncoder, self).__init__()

        # Downsampling layers
        self.downsample1 = DepthwiseConvLayer(
            dim_in=input_channels, dim_out=channels[0], r=2
        )
        self.downsample2 = DepthwiseConvLayer(
            dim_in=channels[0], dim_out=channels[1], r=2
        )
        self.downsample3 = DepthwiseConvLayer(
            dim_in=channels[1], dim_out=channels[2], r=2
        )
        self.downsample4 = DepthwiseConvLayer(
            dim_in=channels[2], dim_out=embed_dim, r=2
        )

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

        self.rec_position_embeddings = nn.Parameter(
            torch.zeros(1, embedding_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        rec_skip_features = []

        x = self.downsample1(x)
        x = self.block1(x)
        rec_skip_features.append(x)

        x = self.downsample2(x)
        x = self.block2(x)
        rec_skip_features.append(x)

        x = self.downsample3(x)
        x = self.block3(x)
        rec_skip_features.append(x)

        x = self.downsample4(x)
        B, C, W, H, Z = x.shape

        x = self.block4(x)
        x = x.flatten(2).transpose(-1, -2)
        x = x + self.rec_position_embeddings
        x = self.dropout(x)

        rec_encoded_shape = (B, C, W, H, Z)

        return x, rec_skip_features, rec_encoded_shape

import torch
import torch.nn as nn

from Encoder import RecEncoder
from Decoder import RecDecoder


class LiST_UNET(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        embed_dim=96,
        embedding_dim=64,
        channels=(24, 48, 60),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        dropout=0.3,
    ):

        super().__init__()

        encoder_args = dict(
            input_channels=in_channels,
            embed_dim=embed_dim,
            embedding_dim=embedding_dim,
            channels=channels,
            blocks=blocks,
            heads=heads,
            r=r,
            dropout=dropout,
        )

        decoder_args = dict(
            rec_channels=out_channels,
            embed_dim=embed_dim,
            channels=channels,
            blocks=blocks,
            heads=heads,
            r=r,
            dropout=dropout,
        )

        self.encoder = RecEncoder(**encoder_args)
        self.decoder = RecDecoder(**decoder_args)

    def forward(self, x):
        encoder_out, skip_features, shape_info = self.encoder(x)
        batch_size, channels, width, height, depth = shape_info
        output = self.decoder(
            encoder_out,
            skip_features,
            (batch_size, channels, width, height, depth),
        )
        return output





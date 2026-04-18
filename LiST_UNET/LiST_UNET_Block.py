import torch.nn as nn
import scipy.io as io


class PatchPartition(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw_pos = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )

    def forward(self, x):
        return self.dw_pos(x)


class LineConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden_channels = channels * 4
        self.expand = nn.Conv3d(
            in_channels=channels,
            out_channels=hidden_channels,
            kernel_size=1,
            bias=False,
        )
        self.non_linear = nn.GELU()
        self.reduce = nn.Conv3d(
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        y = self.expand(x)
        y = self.non_linear(y)
        y = self.reduce(y)
        return y


class LocalRepresentations(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.BatchNorm3d(channels)
        self.proj_in = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )
        self.dw_conv = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.post_norm = nn.BatchNorm3d(channels)
        self.proj_out = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        y = self.pre_norm(x)
        y = self.proj_in(y)
        y = self.dw_conv(y)
        y = self.post_norm(y)
        y = self.proj_out(y)
        return y


class ST_Attention(nn.Module):
    def __init__(self, channels, r, heads):
        super().__init__()
        self.heads = heads
        self.head_channels = channels // heads
        self.scale = self.head_channels ** -0.5

        self.pool = nn.AvgPool3d(kernel_size=1, stride=r)

        self.temporal_qkv = nn.Conv3d(
            in_channels=channels,
            out_channels=channels * 3,
            kernel_size=1,
            bias=False,
        )
        self.spatial_qkv = nn.Conv3d(
            in_channels=channels,
            out_channels=channels * 3,
            kernel_size=1,
            bias=False,
        )

        self.fuse = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        x = self.pool(x)
        b, c, h, w, z = x.shape
        tokens = h * w * z

        t_qkv = self.temporal_qkv(x).view(b, self.heads, -1, tokens)
        q_t, k_t, v_t = t_qkv.split(
            [self.head_channels, self.head_channels, self.head_channels], dim=2
        )
        attn_t = (q_t.transpose(-2, -1) @ k_t).softmax(dim=-1)
        temporal_feat = (v_t @ attn_t).view(b, -1, h, w, z)

        s_qkv = self.spatial_qkv(x).view(b, self.heads, -1, tokens)
        q_s, k_s, v_s = s_qkv.split(
            [self.head_channels, self.head_channels, self.head_channels], dim=2
        )
        attn_s = (q_s @ k_s.transpose(-2, -1)).softmax(dim=-1)
        spatial_feat = (attn_s @ v_s).view(b, -1, h, w, z)

        out = self.fuse(temporal_feat + spatial_feat + x)
        return out


class LocalReverseDiffusion(nn.Module):
    def __init__(self, channels, r):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=r,
            stride=r,
            groups=channels,
        )
        self.channel_norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.channel_mix = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        y = self.upsample(x)
        y = self.channel_norm(y)
        y = self.channel_mix(y)
        return y


class Block(nn.Module):
    def __init__(self, channels, r, heads):
        super().__init__()

        self.pos_embed_1 = PatchPartition(channels)
        self.local_repr = LocalRepresentations(channels)
        self.mlp_1 = LineConv(channels)

        self.pos_embed_2 = PatchPartition(channels)
        self.global_attn = ST_Attention(channels, r, heads)
        self.restore = LocalReverseDiffusion(channels, r)
        self.mlp_2 = LineConv(channels)

    def forward(self, x):
        x = x + self.pos_embed_1(x)
        x = x + self.local_repr(x)
        x = x + self.mlp_1(x)

        x = x + self.pos_embed_2(x)
        x = x + self.restore(self.global_attn(x))
        x = x + self.mlp_2(x)

        return x

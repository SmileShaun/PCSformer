import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath


from .sample import DownSample, UpSample
from .mlp import Mlp


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, h, w):
        super(RelativePositionBias, self).__init__()
        self.num_heads = num_heads
        self.h = int(h)
        self.w = int(w)

        self.relative_position_bias_table = nn.Parameter(
            torch.randn((2 * self.h - 1) * (2 * self.w - 1), self.num_heads) * 0.02)

        coords_h = torch.arange(self.h)
        coords_w = torch.arange(self.w)
        coords = torch.stack(torch.meshgrid(
            [coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.h - 1
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index",
                             relative_position_index)

    def forward(self, H, W):  # H and W is feature map size
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(
            -1)].view(self.h, self.w, self.h * self.w, -1)
        relative_position_bias_expand_h = torch.repeat_interleave(
            relative_position_bias, int(H // self.h), dim=0)
        relative_position_bias_expanded = torch.repeat_interleave(
            relative_position_bias_expand_h, int(W // self.w), dim=1)

        relative_position_bias_expanded = relative_position_bias_expanded.view(int(H * W), self.h * self.w,
                                                                               self.num_heads).permute(2, 0, 1).contiguous().unsqueeze(0)
        return relative_position_bias_expanded


class LocalProxyAttention(nn.Module):
    def __init__(self, dim, reso, num_heads, proxy_downscale, rel_pos, attn_drop_rate, proj_drop_rata):
        super(LocalProxyAttention, self).__init__()
        self.reso = reso
        self.proxy_downscale = proxy_downscale
        self.rel_pos = rel_pos
        self.num_heads = num_heads
        head_dim = dim//num_heads

        if rel_pos:
            self.relative_position_encoding = RelativePositionBias(
                num_heads=num_heads, h=reso//proxy_downscale, w=reso//proxy_downscale)

        self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale.data.fill_(head_dim ** -0.5)

        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rata)

    def forward(self, qkv):
        _, B, L, C = qkv.shape
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.reshape(B, L, self.num_heads, C //
                      self.num_heads).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, L, self.num_heads, C //
                      self.num_heads).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, L, self.num_heads, C //
                      self.num_heads).permute(0, 2, 1, 3).contiguous()
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(
                self.reso/self.proxy_downscale, self.reso/self.proxy_downscale)
            attn = attn + relative_position_bias
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, L, C)

        x = self.proj_drop(self.proj(x))

        return x


class LocalProxyTransformer(nn.Module):
    def __init__(self, dim, reso, proxy_downscale, num_heads, network_depth, mlp_ratio, rel_pos, qkv_bias, attn_drop_rate, proj_drop_rata, norm_layer):
        super(LocalProxyTransformer, self).__init__()

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.avgpool = nn.AvgPool2d(kernel_size=proxy_downscale)
        self.maxpool = nn.MaxPool2d(kernel_size=proxy_downscale)
        self.downsample = DownSample(
            in_channels=dim, out_channels=dim, patch_size=proxy_downscale, kernel_size=None)
        self.upsample = UpSample(
            in_channels=dim, out_channels=dim, patch_size=proxy_downscale, kernel_size=None)

        self.gate = nn.Conv2d(in_channels=dim*3, out_channels=3,
                              kernel_size=3, stride=1, padding=1, bias=True)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(in_dim=dim, network_depth=network_depth,
                       hidden_dim=mlp_hidden_dim, out_dim=dim)

        self.attn = LocalProxyAttention(dim=dim, reso=reso, num_heads=num_heads, proxy_downscale=proxy_downscale,
                                        rel_pos=rel_pos, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata)

    def seq2img(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        return x

    def img2seq(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1).contiguous()
        return x

    def forward(self, x):
        B, L, C = x.shape
        identity = x

        avg_proxy = self.avgpool(self.seq2img(x))
        max_proxy = self.maxpool(self.seq2img(x))
        conv_proxy = self.downsample(self.seq2img(x))
        gates = self.gate(torch.cat((avg_proxy, max_proxy, conv_proxy), dim=1))
        x = avg_proxy * gates[:, [0], :, :] + max_proxy * \
            gates[:, [1], :, :] + conv_proxy * gates[:, [2], :, :]
        x = self.img2seq(x)

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3).contiguous()
        attened_x = self.attn(self.norm1(qkv))

        attened_x = self.upsample(self.seq2img(attened_x))
        x = identity + self.img2seq(attened_x)

        x = x + self.mlp(self.norm2(x))

        return x

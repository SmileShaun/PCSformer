import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import DropPath

from .mlp import Mlp


class Attention(nn.Module):
    def __init__(self, dim, resolution, stripeWidth, attention_mode, num_heads, currentDepth, attn_drop_rate, proj_drop_rata):
        super(Attention, self).__init__()
        self.resolution = resolution
        self.stripeWidth = stripeWidth
        self.attention_mode = attention_mode
        self.resolution = resolution
        self.num_heads = num_heads
        self.head_dim = dim//num_heads
        self.currentDepth = currentDepth

        self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale.data.fill_(self.head_dim ** -0.5)

        window_sizes = {
            -1: (resolution, resolution),
            0: (resolution, stripeWidth),
            1: (stripeWidth, resolution)
        }
        self.windowHeight, self.windowWidth = window_sizes.get(
            attention_mode, None)
        if self.windowHeight is None or self.windowWidth is None:
            print("ERROR MODE", attention_mode)

        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj_drop = nn.Dropout(proj_drop_rata)
        self.proj = nn.Linear(dim, dim)
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3,
                               stride=1, padding=1, groups=dim)

    def seq2CSwin(self, x):
        B, L, C = x.shape
        H = W = int(np.sqrt(L))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        x = x.view(B, C, H//self.windowHeight,
                   self.windowHeight, W//self.windowWidth, self.windowWidth).permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, self.windowHeight*self.windowWidth, C)
        x = x.reshape(-1, self.windowHeight*self.windowWidth,
                      self.num_heads, C//self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def CSwin2img(self, x, windowHeight, windowWidth, resolution):
        B = int(x.shape[0]/(resolution*resolution/windowHeight/windowWidth))
        x = x.view(B, resolution//windowHeight, resolution //
                   windowWidth, windowHeight, windowWidth, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            B, resolution, resolution, -1)
        return x

    def shiftedWindowVerticalPartition(self, x, reso, stripes):
        B, H, W, C = x.shape
        x = x.view(B, H//reso, reso, W//stripes, stripes, C)
        windows = x.permute(
            0, 1, 3, 2, 4, 5).contiguous().view(-1, reso*stripes, C)
        return windows

    def shiftedWindowHorizontalPartition(self, x, reso, stripes):
        B, H, W, C = x.shape
        x = x.view(B, H//stripes, stripes, W//reso, reso, C)
        windows = x.permute(
            0, 1, 3, 2, 4, 5).contiguous().view(-1, reso*stripes, C)
        return windows

    def verticalAttnMask(self, resolution, stripeWidth):
        shift_size = int(stripeWidth//2)
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        img_mask = torch.zeros((1, resolution, resolution, 1), device=device)
        w_slices = (slice(0, -stripeWidth),
                    slice(-stripeWidth, -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for w in w_slices:
            img_mask[:, :, w, :] = cnt
            cnt += 1

        mask_windows = self.shiftedWindowVerticalPartition(
            img_mask, reso=resolution, stripes=stripeWidth)
        mask_windows = mask_windows.view(-1, resolution*stripeWidth)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def horizontalMask(self, resolution, stripeWidth):
        shift_size = int(stripeWidth//2)

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        img_mask = torch.zeros((1, resolution, resolution, 1), device=device)
        h_slices = (slice(0, -stripeWidth),
                    slice(-stripeWidth, -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            img_mask[:, h, :, :] = cnt
            cnt += 1

        mask_windows = self.shiftedWindowHorizontalPartition(
            img_mask, reso=resolution, stripes=stripeWidth)
        mask_windows = mask_windows.view(-1, stripeWidth*resolution)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def get_lepe(self, x, func):
        B, L, C = x.shape
        H = W = int(np.sqrt(L))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.windowHeight, self.windowWidth
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)

        lepe = func(x)
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads,
                            H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        x = x.reshape(-1, self.num_heads, C // self.num_heads,
                      self.windowHeight * self.windowWidth).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Check size
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # [B*(H//windowHeight)*(W//windowWidth), num_heads, windowHeight*windowWidth, C//num_heads]
        q, k = self.seq2CSwin(q), self.seq2CSwin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.currentDepth % 2 == 1:
            if self.attention_mode == 0:
                mask = self.verticalAttnMask(self.resolution, self.stripeWidth)
            elif self.attention_mode == 1:
                mask = self.horizontalMask(self.resolution, self.stripeWidth)
            mask = mask.repeat(B, 1, 1)
            mask = mask.unsqueeze(1)
            attn = attn + mask
            attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        else:
            attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.windowHeight *
                                      self.windowWidth, C)
        x = self.proj_drop(self.proj(x))
        x = self.CSwin2img(x, windowHeight=self.windowHeight,
                           windowWidth=self.windowWidth, resolution=self.resolution).view(B, -1, C)

        return x


class SlidingCrossStripeTransformer(nn.Module):
    def __init__(self, dim, resolution, num_heads, stripeWidth, network_depth, mlp_ratio, qkv_bias, attn_drop_rate, proj_drop_rata, drop_path, norm_layer, currentDepth):
        super(SlidingCrossStripeTransformer, self).__init__()
        self.resolution = resolution
        self.currentDepth = currentDepth
        self.stripeWidth = stripeWidth

        self.no_cross_stripes = resolution == stripeWidth
        self.branch_num = 1 if self.no_cross_stripes else 2

        if self.no_cross_stripes:
            self.attns = nn.ModuleList([
                Attention(dim=dim, resolution=resolution, stripeWidth=stripeWidth, attention_mode=-1,
                          num_heads=num_heads, currentDepth=0, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata)
                for i in range(self.branch_num)
            ])
        else:
            self.attns = nn.ModuleList([
                Attention(dim=dim//2, resolution=resolution, stripeWidth=stripeWidth, attention_mode=i,
                          num_heads=num_heads//2, currentDepth=currentDepth, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata)
                for i in range(self.branch_num)
            ])

        mlp_hidden_dim = int(dim*mlp_ratio)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_dim=dim, network_depth=network_depth,
                       hidden_dim=mlp_hidden_dim, out_dim=dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)

    def shiftedWindowPartition(self, x):
        B, L, C = x.shape[1:]
        x = x.reshape(3, B, L//self.resolution, L//self.resolution, C)
        shift_size = self.stripeWidth//2
        x = torch.roll(
            x, shifts=(-shift_size, -shift_size), dims=(2, 3))
        x = x.reshape(3, B, -1, C)
        return x

    def reverseShiftedWindowPartition(self, x):
        B, L, C = x.shape
        x = x.reshape(B, L//self.resolution, L//self.resolution, C)
        shift_size = self.stripeWidth//2
        x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        H = W = self.resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        qkv = self.qkv(x).reshape(
            B, -1, 3, C).permute(2, 0, 1, 3).contiguous()

        if self.currentDepth % 2 == 1 and self.no_cross_stripes == False:
            qkv = self.shiftedWindowPartition(qkv)

        qkv = self.norm1(qkv)

        attened_x = (
            torch.cat([self.attns[0](qkv[:, :, :, :C//2]),
                      self.attns[1](qkv[:, :, :, C//2:])], dim=2)
            if self.branch_num == 2
            else self.attns[0](qkv)
        )

        if self.currentDepth % 2 == 1 and self.no_cross_stripes == False:
            attened_x = self.reverseShiftedWindowPartition(attened_x)

        x = x + self.drop_path(attened_x)

        x = x + self.drop_path((self.mlp(self.norm2(x))))
        return x

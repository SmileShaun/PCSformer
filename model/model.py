import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_


from utils import blc_2_bchw,  bchw_2_blc
from .PatchEmbed import PatchEmbed
from .SlidingCrossStripeTransformer import SlidingCrossStripeTransformer
from .LocalProxyTransformer import LocalProxyTransformer
from .sample import DownSample, UpSample
from .RefinementBlock import RefinementBlock


class PCSformer(nn.Module):
    def __init__(self, config):
        super(PCSformer, self).__init__()
        model_resolution = config['model']['model_resolution']
        in_channels = config['model']['in_channels']
        embed_dim = config['model']['embed_dim']
        depth = config['model']['depth']
        split_size = config['model']['split_size']
        proxy_downscale = config['model']['proxy_downscale']
        num_heads = config['model']['num_heads']
        mlp_ratio = config['model']['mlp_ratio']
        qkv_bias = config['model']['qkv_bias']
        attn_drop_rate = config['model']['attn_drop_rate']
        proj_drop_rata = config['model']['proj_drop_rata']
        drop_path_rate = config['model']['drop_path_rate']
        num_refinement_blocks = config['model']['num_refinement_blocks']
        refinement_block_dim = config['model']['refinement_block_dim']
        norm_layer = nn.LayerNorm

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                np.sum(depth))]  # stochastic depth decay rule

        self.patch_embed = PatchEmbed(
            in_channels, embed_dim[0], kernel_size=[7, 3], patch_size=2)

        self.stage1 = nn.ModuleList([
            SlidingCrossStripeTransformer(dim=embed_dim[0], resolution=model_resolution//2, num_heads=num_heads[0],
                                          stripeWidth=split_size[0], network_depth=sum(depth), mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata,
                                          drop_path=dpr[i], norm_layer=nn.LayerNorm, currentDepth=i)
            for i in range(depth[0])
        ])
        self.LocalProxyTransformer1 = LocalProxyTransformer(
            dim=embed_dim[0], reso=model_resolution//2, proxy_downscale=proxy_downscale[0], num_heads=num_heads[0], network_depth=sum(depth),
            mlp_ratio=mlp_ratio, rel_pos=True, qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata, norm_layer=norm_layer)
        self.merge1 = DownSample(
            in_channels=embed_dim[0], out_channels=embed_dim[1], patch_size=2, kernel_size=5)
        self.stage1_3 = nn.Sequential(nn.Conv2d(in_channels=embed_dim[0], out_channels=embed_dim[0], kernel_size=4, stride=4, groups=embed_dim[0]),
                                      nn.Conv2d(in_channels=embed_dim[0], out_channels=embed_dim[0], kernel_size=1))
        self.stage1_4 = nn.Sequential(nn.Conv2d(in_channels=embed_dim[0], out_channels=embed_dim[0], kernel_size=2, stride=2, groups=embed_dim[0]),
                                      nn.Conv2d(in_channels=embed_dim[0], out_channels=embed_dim[0], kernel_size=1))

        self.stage2 = nn.ModuleList([
            SlidingCrossStripeTransformer(dim=embed_dim[1], resolution=model_resolution//4, num_heads=num_heads[1],
                                          stripeWidth=split_size[1], network_depth=sum(depth), mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata,
                                          drop_path=dpr[np.sum(depth[:1])+i], norm_layer=nn.LayerNorm, currentDepth=i)
            for i in range(depth[1])
        ])
        self.LocalProxyTransformer2 = LocalProxyTransformer(
            dim=embed_dim[1], reso=model_resolution//4, proxy_downscale=proxy_downscale[1], num_heads=num_heads[1], network_depth=sum(depth),
            mlp_ratio=mlp_ratio, rel_pos=True, qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata, norm_layer=norm_layer)
        self.merge2 = DownSample(
            in_channels=embed_dim[1], out_channels=embed_dim[2], patch_size=2, kernel_size=3)
        self.stage2_5 = nn.Sequential(UpSample(in_channels=embed_dim[1], out_channels=embed_dim[1], patch_size=2, kernel_size=None),
                                      nn.Conv2d(embed_dim[1], embed_dim[0]//2, kernel_size=1))

        self.stage3 = nn.ModuleList([
            SlidingCrossStripeTransformer(dim=embed_dim[2], resolution=model_resolution//8, num_heads=num_heads[2],
                                          stripeWidth=split_size[2], network_depth=sum(depth), mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata,
                                          drop_path=dpr[np.sum(depth[:2])+i], norm_layer=nn.LayerNorm, currentDepth=i)
            for i in range(depth[2])
        ])
        self.LocalProxyTransformer3 = LocalProxyTransformer(
            dim=embed_dim[2], reso=model_resolution//8, proxy_downscale=proxy_downscale[2], num_heads=num_heads[2], network_depth=sum(depth),
            mlp_ratio=mlp_ratio, rel_pos=True, qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata, norm_layer=norm_layer)
        self.upMerge1 = UpSample(
            in_channels=embed_dim[2], out_channels=embed_dim[3], patch_size=2, kernel_size=None)
        self.stage3_5 = nn.Sequential(UpSample(in_channels=embed_dim[2], out_channels=embed_dim[2], patch_size=4, kernel_size=None),
                                      nn.Conv2d(embed_dim[2], embed_dim[0]//2, kernel_size=1))

        self.stage4 = nn.ModuleList([
            SlidingCrossStripeTransformer(dim=embed_dim[3], resolution=model_resolution//4, num_heads=num_heads[3],
                                          stripeWidth=split_size[3], network_depth=sum(depth), mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata,
                                          drop_path=dpr[np.sum(depth[:3])+i], norm_layer=nn.LayerNorm, currentDepth=i)
            for i in range(depth[3])
        ])
        self.upMerge2 = UpSample(
            in_channels=embed_dim[3], out_channels=embed_dim[4], patch_size=2, kernel_size=None)

        self.stage5 = nn.ModuleList([
            SlidingCrossStripeTransformer(dim=embed_dim[4], resolution=model_resolution//2, num_heads=num_heads[4],
                                          stripeWidth=split_size[4], network_depth=sum(depth), mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata,
                                          drop_path=dpr[np.sum(depth[:4])+i], norm_layer=nn.LayerNorm, currentDepth=i)
            for i in range(depth[4])
        ])
        self.lastUpMetge = UpSample(
            in_channels=embed_dim[4], out_channels=embed_dim[4], patch_size=2, kernel_size=3)

        self.conv = nn.Conv2d(
            embed_dim[4], 4, kernel_size=3, stride=1, padding=1)

        self.proj3 = nn.Conv2d(
            embed_dim[0]+embed_dim[2], embed_dim[2], kernel_size=1)
        self.proj4 = nn.Conv2d(
            embed_dim[0]+embed_dim[1]+embed_dim[3], embed_dim[3], kernel_size=1)
        self.proj5 = nn.Conv2d(
            2*embed_dim[0]+embed_dim[4], embed_dim[4], kernel_size=1)

        self.refineProj1 = nn.Conv2d(
            3, refinement_block_dim, kernel_size=3, padding=1)
        self.refine_blocks = nn.ModuleList([
            RefinementBlock(dim=refinement_block_dim, kernel_size=3)
            for _ in range(num_refinement_blocks)
        ])
        self.refineProj2 = nn.Conv2d(
            refinement_block_dim, 3, kernel_size=3, padding=1)

    def forward_features(self, x):
        x = self.patch_embed(x)

        x1, x2 = x, x
        for blk in self.stage1:
            x1 = blk(x1)
        x2 = self.LocalProxyTransformer1(x2)
        x = blc_2_bchw(x1 + x2)
        skip1_3, skip1_4, skip1_5 = self.stage1_3(x), self.stage1_4(x), x

        x = bchw_2_blc(self.merge1(x))
        x1, x2 = x, x
        for blk in self.stage2:
            x1 = blk(x1)
        x2 = self.LocalProxyTransformer2(x2)
        x = blc_2_bchw(x1 + x2)
        skip2_5, skip2_4 = self.stage2_5(x), x

        x = bchw_2_blc(self.proj3(torch.cat((self.merge2(x), skip1_3), dim=1)))
        x1, x2 = x, x
        for blk in self.stage3:
            x1 = blk(x1)
        x2 = self.LocalProxyTransformer3(x2)
        x = blc_2_bchw(x1 + x2)
        skip3_5 = self.stage3_5(x)

        x = bchw_2_blc(self.proj4(
            torch.cat((self.upMerge1(x), skip1_4, skip2_4), dim=1)))
        for blk in self.stage4:
            x = blk(x)
        x = blc_2_bchw(x)

        x = bchw_2_blc(self.proj5(
            torch.cat((self.upMerge2(x), skip1_5, skip2_5, skip3_5), dim=1)))
        for blk in self.stage5:
            x = blk(x)

        return self.conv(self.lastUpMetge(blc_2_bchw(x)))

    def RefineNetwork(self, x):
        short_cut = x
        x = self.refineProj1(x)
        for block in self.refine_blocks:
            x = block(x)
        return short_cut + self.refineProj2(x)

    def forward(self, x):
        H, W = x.shape[2:]
        feat = self.forward_features(x)
        K, B = torch.split(feat, (1, 3), dim=1)

        x = K * x - B + x
        coarseDehazedImage = x[:, :, :H, :W]
        refinedDehazedImage = self.RefineNetwork(coarseDehazedImage)
        return refinedDehazedImage

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SwinBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2,1,kernel_size=7,padding=3),
            nn.Sigmoid()
        )
    def forward(self, x):
        ca = self.ca(x) * x
        max_pool = torch.max(ca, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(ca, dim=1, keepdim=True)
        sa = self.spatial(torch.cat([avg_pool, max_pool], dim=1)) * ca
        return sa

# Advanced Swin-like UNet
class AdvancedSwinUNet(nn.Module):
    def __init__(self, in_ch=18, out_ch=1, embed_dim=120, depth=3, heads=5, img_size=(256,256)):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch = nn.Conv2d(in_ch, embed_dim, kernel_size=4, stride=4, padding=0)  # reduce spatial resolution by 4
        # simple positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size[0]//4)*(img_size[1]//4), embed_dim))
        self.encoder = nn.ModuleList([SwinBlock(embed_dim, heads=heads) for _ in range(depth)])
        # residual convs applied at each scale (we keep single scale for simplicity)
        self.res_conv = nn.ModuleList([ResidualConvBlock(embed_dim, embed_dim) for _ in range(depth)])
        self.cbam = CBAM(embed_dim, reduction=8)
        # decoder (UNet-like upsampling with residual conv blocks)
        self.up1 = nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=2, stride=2)
        self.dec1 = ResidualConvBlock(embed_dim//2, embed_dim//2)
        self.up2 = nn.ConvTranspose2d(embed_dim//2, embed_dim//4, kernel_size=2, stride=2)
        self.dec2 = ResidualConvBlock(embed_dim//4, embed_dim//4)
        self.final = nn.Conv2d(embed_dim//4, out_ch, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W], assume H,W divisible by 4
        B, C, H, W = x.shape
        p = self.patch(x)                      # [B, E, H/4, W/4]
        Hp, Wp = p.shape[2], p.shape[3]
        z = p.flatten(2).transpose(1,2)        # [B, N, E]
        
        # add pos embed (resize if different) - VERSION EXACTE DE VOTRE CODE
        if self.pos_embed.shape[1] != z.shape[1]:
            # interpolate pos_embed spatially
            pe = self.pos_embed[0].transpose(0,1).view(self.embed_dim, int(np.sqrt(self.pos_embed.shape[1])), -1)
            # fallback: zeros (comme dans votre code original)
            z = z + 0
        else:
            z = z + self.pos_embed

        # transformer blocks + conv residuals
        for blk, res in zip(self.encoder, self.res_conv):
            z = blk(z)                          # [B,N,E]
            zp = z.transpose(1,2).view(B, self.embed_dim, Hp, Wp)
            zp = res(zp)
            z = zp.flatten(2).transpose(1,2)

        # bring back to spatial
        feat = z.transpose(1,2).view(B, self.embed_dim, Hp, Wp)
        feat = self.cbam(feat)

        # decoder: upsample twice to original resolution (factor 4 total)
        d1 = self.up1(feat)                    # [B, E/2, Hp*2, Wp*2]
        d1 = self.dec1(d1)
        d2 = self.up2(d1)                      # [B, E/4, Hp*4, Wp*4] == original H,W
        d2 = self.dec2(d2)
        out = self.final(d2)
        return out

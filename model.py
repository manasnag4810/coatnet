# Model architecture placeholder
import torch
import torch.nn as nn
import torch.nn.functional as F


class MBConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_factor=4):
        super(MBConvBlock, self).__init__()
        mid_ch = in_ch * expansion_factor
        self.expand = nn.Conv3d(in_ch, mid_ch, kernel_size=1)
        self.depthwise = nn.Conv3d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch)
        self.project = nn.Conv3d(mid_ch, out_ch, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(mid_ch)
        self.bn2 = nn.BatchNorm3d(mid_ch)
        self.bn3 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.expand(x)))
        x = self.relu(self.bn2(self.depthwise(x)))
        x = self.bn3(self.project(x))
        return x + identity


class RelativeSelfAttention(nn.Module):
    def __init__(self, dim):
        super(RelativeSelfAttention, self).__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.transpose(0, 1), qkv)  # (N, B, C)
        attn = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(0, 1)
        out = self.proj(out)
        out = out.permute(0, 2, 1).view(B, C, D, H, W)
        return out


class MLDRBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MLDRBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=3, dilation=3)
        self.conv3 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=5, dilation=5)
        self.bn = nn.BatchNorm3d(out_ch * 3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.relu(self.bn(out))
        return out


class CoAtUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super(CoAtUNet, self).__init__()
        self.enc1 = MBConvBlock(in_ch, base_ch)
        self.enc2 = MBConvBlock(base_ch, base_ch * 2)
        self.attn1 = RelativeSelfAttention(base_ch * 2)
        self.attn2 = RelativeSelfAttention(base_ch * 2)

        self.pool = nn.MaxPool3d(2)

        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = MLDRBlock(base_ch * 2, base_ch)

        self.up2 = nn.ConvTranspose3d(base_ch, base_ch // 2, kernel_size=2, stride=2)
        self.dec2 = MLDRBlock(base_ch, base_ch // 2)

        self.final = nn.Conv3d(base_ch // 2 * 3, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.pool(e1)
        e2 = self.enc2(e2)
        e2 = self.attn1(e2)
        e2 = self.attn2(e2)

        d1 = self.up1(e2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, x], dim=1))

        out = torch.sigmoid(self.final(d2))
        return out

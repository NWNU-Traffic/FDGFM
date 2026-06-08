import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v: float, divisor: int = 8) -> int:
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else (k[0] // 2, k[1] // 2)
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DualPoolChannelGate(nn.Module):
    """ECA-like channel gate using both average and max pooled descriptors."""
    def __init__(self, channels: int, b: int = 1, gamma: int = 2):
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 else k + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        mx = self.max_pool(x).squeeze(-1).transpose(-1, -2)
        w = self.conv(avg) + self.conv(mx)
        w = self.act(w).transpose(-1, -2).unsqueeze(-1)
        return x * w


class PatchSpectralEnhancer(nn.Module):
    """Submodule 1: gated local FFN + patch-wise Fourier recalibration.

    It strengthens weak high-frequency edges and shape boundaries. Padding is
    handled internally, so H and W do not need to be divisible by patch_size.
    """
    def __init__(self, channels: int, patch_size: int = 8, expansion: float = 1.0):
        super().__init__()
        hidden = _make_divisible(channels * expansion, 8)
        self.patch_size = patch_size
        self.project_in = nn.Conv2d(channels, hidden * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.freq_weight = nn.Parameter(torch.ones(1, channels, 1, 1, patch_size, patch_size // 2 + 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def _pad_to_patch(self, x):
        _, _, h, w = x.shape
        p = self.patch_size
        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, h, w

    def forward(self, x):
        shortcut = x
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        x, h0, w0 = self._pad_to_patch(x)
        b, c, h, w = x.shape
        p = self.patch_size
        # [B,C,H,W] -> [B,C,H/P,W/P,P,P]
        x_patch = x.view(b, c, h // p, p, w // p, p).permute(0, 1, 2, 4, 3, 5).contiguous()
        x_fft = torch.fft.rfft2(x_patch.float(), dim=(-2, -1), norm='ortho')
        x_fft = x_fft * self.freq_weight
        x_patch = torch.fft.irfft2(x_fft, s=(p, p), dim=(-2, -1), norm='ortho')
        x = x_patch.permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, h, w)
        x = x[:, :, :h0, :w0]
        x = self.bn(x)
        return shortcut + self.gamma * x


class OrientedDepthwiseContext(nn.Module):
    """Submodule 2: multi-directional depthwise context aggregation.

    The asymmetric kernels model long horizontal/vertical sign boundaries and
    road-scene context with very few parameters.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.stem = ConvBNAct(channels, channels, k=3, g=channels)
        self.b1 = nn.Sequential(
            ConvBNAct(channels, channels, k=(1, 7), g=channels),
            ConvBNAct(channels, channels, k=(7, 1), g=channels),
        )
        self.b2 = nn.Sequential(
            ConvBNAct(channels, channels, k=(1, 11), g=channels),
            ConvBNAct(channels, channels, k=(11, 1), g=channels),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=3, dilation=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.proj = ConvBNAct(channels, channels, k=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        shortcut = x
        x0 = self.stem(x)
        y = x0 + self.b1(x0) + self.b2(x0) + self.b3(x0)
        avg = torch.mean(y, dim=1, keepdim=True)
        mx, _ = torch.max(y, dim=1, keepdim=True)
        y = y * self.spatial_gate(torch.cat([avg, mx], dim=1))
        y = self.proj(y)
        return shortcut + self.gamma * y


class AdaptiveBranchFusion(nn.Module):
    """Submodule 3: spatially adaptive softmax fusion + channel recalibration."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, 3, 1, bias=True)
        )
        self.channel_gate = DualPoolChannelGate(channels)
        self.out_proj = ConvBNAct(channels, channels, k=1)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, spectral, context):
        logits = self.gate(torch.cat([x, spectral, context], dim=1))
        weights = torch.softmax(logits, dim=1)
        y = weights[:, 0:1] * x + weights[:, 1:2] * spectral + weights[:, 2:3] * context
        y = self.channel_gate(y)
        y = self.out_proj(y)
        return x + self.beta * y


class FDGFM(nn.Module):
    """Frequency-guided Directional Gated Fusion Module.

    Recommended insertion: after C3/C4/C5 projection layers in RT-DETR-R18 or
    after P3/P4/P5 neck features in YOLO-style detectors.
    """
    def __init__(self, channels: int, patch_size: int = 8, expansion: float = 1.0, reduction: int = 8):
        super().__init__()
        self.spectral = PatchSpectralEnhancer(channels, patch_size, expansion)
        self.context = OrientedDepthwiseContext(channels)
        self.fusion = AdaptiveBranchFusion(channels, reduction)

    def forward(self, x):
        xs = self.spectral(x)
        xc = self.context(x)
        return self.fusion(x, xs, xc)


if __name__ == '__main__':
    for c, h, w in [(128, 80, 80), (256, 40, 40), (512, 20, 20), (256, 37, 61)]:
        m = FDGFM(c, patch_size=8, expansion=1.0)
        x = torch.randn(2, c, h, w)
        y = m(x)
        n = sum(p.numel() for p in m.parameters())
        print(f'C={c}, input={tuple(x.shape)}, output={tuple(y.shape)}, params={n/1e6:.3f}M')

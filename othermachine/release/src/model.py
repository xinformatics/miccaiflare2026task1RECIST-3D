"""Lean prompt-conditioned 3D U-Net for RECIST->3D lesion segmentation."""
import torch, torch.nn as nn, torch.nn.functional as F

def norm(c):
    return nn.InstanceNorm3d(c, affine=True)

class ConvBlock(nn.Module):
    def __init__(self, ci, co, n=2, stride=1):
        super().__init__()
        layers = []
        for i in range(n):
            layers += [nn.Conv3d(ci if i == 0 else co, co, 3, stride=stride if i == 0 else 1,
                                 padding=1, bias=False), norm(co), nn.LeakyReLU(0.01, inplace=True)]
        self.b = nn.Sequential(*layers)
    def forward(self, x): return self.b(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=4, base=16, levels=5, out_ch=1, deep_sup=3, max_ch=320):
        super().__init__()
        chs = [min(base * 2 ** i, max_ch) for i in range(levels)]
        self.chs = chs
        self.enc = nn.ModuleList()
        self.enc.append(ConvBlock(in_ch, chs[0], 2, 1))
        for i in range(1, levels):
            self.enc.append(ConvBlock(chs[i - 1], chs[i], 2, 2))
        self.up = nn.ModuleList(); self.dec = nn.ModuleList()
        for i in range(levels - 1, 0, -1):
            self.up.append(nn.ConvTranspose3d(chs[i], chs[i - 1], 2, 2))
            self.dec.append(ConvBlock(chs[i - 1] * 2, chs[i - 1], 2, 1))
        self.deep_sup = deep_sup
        self.heads = nn.ModuleList([nn.Conv3d(chs[i], out_ch, 1) for i in range(deep_sup)])

    def forward(self, x):
        feats = []
        for e in self.enc:
            x = e(x); feats.append(x)
        outs = []
        x = feats[-1]
        for k, (u, d) in enumerate(zip(self.up, self.dec)):
            x = u(x)
            skip = feats[-2 - k]
            x = d(torch.cat([x, skip], 1))
            lvl = len(self.chs) - 2 - k
            if lvl < self.deep_sup:
                outs.append(self.heads[lvl](x))
        outs = outs[::-1]              # finest first
        return outs if self.training else outs[0]

# ---- prompt channels, built on-device from RECIST endpoints -------------------

def build_input(img_u8, p1, p2, diam_mm, s_mm, shape):
    """img_u8: (B,Z,Y,X) uint8/float; p1,p2: (B,3) ROI voxel coords (z,y,x);
    diam_mm,s_mm: (B,). -> (B,4,Z,Y,X) float32."""
    B = img_u8.shape[0]; Z, Y, X = shape
    dev = img_u8.device
    zz = torch.arange(Z, device=dev, dtype=torch.float32).view(1, Z, 1, 1)
    yy = torch.arange(Y, device=dev, dtype=torch.float32).view(1, 1, Y, 1)
    xx = torch.arange(X, device=dev, dtype=torch.float32).view(1, 1, 1, X)
    a = p1.view(B, 3, 1, 1, 1); b = p2.view(B, 3, 1, 1, 1)
    ab = b - a
    ab2 = (ab ** 2).sum(1, keepdim=True).clamp_min(1e-6)
    pz = zz - a[:, 0]; py = yy - a[:, 1]; px = xx - a[:, 2]
    t = ((pz * ab[:, 0] + py * ab[:, 1] + px * ab[:, 2]) / ab2[:, 0]).clamp(0, 1)
    dz = pz - t * ab[:, 0]; dy = py - t * ab[:, 1]; dx = px - t * ab[:, 2]
    dist_vox = torch.sqrt(dz * dz + dy * dy + dx * dx + 1e-8)
    dist_mm = dist_vox * s_mm.view(B, 1, 1, 1)
    line = (dist_vox <= 1.2).float()
    prior = torch.exp(-dist_mm / (0.4 * diam_mm.view(B, 1, 1, 1)))
    scale = torch.log(diam_mm / 30.0).view(B, 1, 1, 1).expand(B, Z, Y, X) / 1.5
    img = img_u8.float() / 255.0
    return torch.stack([img, line, prior, scale], 1)

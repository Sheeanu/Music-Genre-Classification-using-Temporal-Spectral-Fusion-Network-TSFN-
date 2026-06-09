import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.GELU(),
            nn.Linear(channels // ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class FeatureStream(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

        self.se = SEBlock(out_dim)

    def forward(self, x):
        return self.se(self.net(x))


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()

        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, a, b):

        B, D = a.shape
        H = self.heads
        Dh = D // H

        q = self.q_proj(a).view(B, H, 1, Dh)
        k = self.k_proj(b).view(B, H, 1, Dh)
        v = self.v_proj(b).view(B, H, 1, Dh)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).view(B, D)

        return self.norm(a + self.out(out))


class TSFN(nn.Module):
    def __init__(self, feat_dim, n_classes=10):
        super().__init__()

        self.split = feat_dim // 2

        self.stream1 = FeatureStream(self.split)
        self.stream2 = FeatureStream(feat_dim - self.split)

        self.cross_ab = CrossAttentionFusion()
        self.cross_ba = CrossAttentionFusion()

        self.gate = nn.Sequential(
            nn.Linear(256, 2),
            nn.Softmax(dim=-1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, n_classes)
        )

    def forward(self, x):

        x1 = x[:, :self.split]
        x2 = x[:, self.split:]

        s1 = self.stream1(x1)
        s2 = self.stream2(x2)

        s1 = self.cross_ab(s1, s2)
        s2 = self.cross_ba(s2, s1)

        combined = torch.cat([s1, s2], dim=-1)

        return self.classifier(combined)

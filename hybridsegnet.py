import torch
import torch.nn as nn
import torch.nn.functional as F

from lightweight_vit import LightViTEncoder
from GAB import GAB
from SS2D import VSSMBlock


class PatchExpand(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.expand(x)


class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = PatchExpand(in_channels, out_channels)
        self.skip_gab = GAB(skip_channels)
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1)

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.vssm = VSSMBlock(dim=out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        skip = self.skip_gab(skip)
        skip = self.skip_proj(skip)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        x = self.fusion(x)
        x = self.vssm(x)

        return x


class HybridViTGABVSSMUNet(nn.Module):
    """
    Hybrid ViT-GAB-VSSM segmentation network with deep supervision.

    Training output:
        final_out, aux3, aux2, aux1

    Inference output:
        final_out
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        encoder_dims=(32, 64, 128, 256),
        encoder_depths=(2, 2, 2, 2),
        encoder_heads=(1, 2, 4, 8),
        dropout=0.1,
        deep_supervision=True
    ):
        super().__init__()

        self.deep_supervision = deep_supervision

        c1, c2, c3, c4 = encoder_dims

        self.encoder = LightViTEncoder(
            in_channels=in_channels,
            dims=encoder_dims,
            depths=encoder_depths,
            heads=encoder_heads,
            dropout=dropout
        )

        self.bottleneck = nn.Sequential(
            VSSMBlock(dim=c4),
            VSSMBlock(dim=c4)
        )

        self.decoder3 = DecoderStage(
            in_channels=c4,
            skip_channels=c3,
            out_channels=c3
        )

        self.decoder2 = DecoderStage(
            in_channels=c3,
            skip_channels=c2,
            out_channels=c2
        )

        self.decoder1 = DecoderStage(
            in_channels=c2,
            skip_channels=c1,
            out_channels=c1
        )

        self.final_up = nn.Sequential(
            PatchExpand(c1, c1),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        self.seg_head = nn.Conv2d(c1, num_classes, kernel_size=1)

        # Deep supervision heads
        self.aux_head3 = nn.Conv2d(c3, num_classes, kernel_size=1)
        self.aux_head2 = nn.Conv2d(c2, num_classes, kernel_size=1)
        self.aux_head1 = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]

        f1, f2, f3, f4 = self.encoder(x)

        x = self.bottleneck(f4)

        d3 = self.decoder3(x, f3)
        d2 = self.decoder2(d3, f2)
        d1 = self.decoder1(d2, f1)

        out = self.final_up(d1)

        if out.shape[-2:] != input_size:
            out = F.interpolate(
                out,
                size=input_size,
                mode="bilinear",
                align_corners=False
            )

        final_out = self.seg_head(out)

        if self.deep_supervision and self.training:
            aux3 = self.aux_head3(d3)
            aux2 = self.aux_head2(d2)
            aux1 = self.aux_head1(d1)

            aux3 = F.interpolate(aux3, size=input_size, mode="bilinear", align_corners=False)
            aux2 = F.interpolate(aux2, size=input_size, mode="bilinear", align_corners=False)
            aux1 = F.interpolate(aux1, size=input_size, mode="bilinear", align_corners=False)

            return final_out, aux3, aux2, aux1

        return final_out

"""
if __name__ == "__main__":
    model = HybridViTGABVSSMUNet(
        in_channels=3,
        num_classes=1,
        encoder_dims=(32, 64, 128, 256),
        deep_supervision=True
    )

    x = torch.randn(2, 3, 256, 256)

    model.train()
    outputs = model(x)

    print("Training outputs:")
    for i, out in enumerate(outputs):
        print(f"Output {i}: {out.shape}")

    model.eval()
    with torch.no_grad():
        y = model(x)

    print("Inference output:", y.shape)
"""    
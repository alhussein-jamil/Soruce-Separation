import torch
import torch.nn as nn


class MaskUNet(nn.Module):
    """U-Net that predicts a multiplicative mask over the input spectrogram."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.enc_act = nn.LeakyReLU(0.2, inplace=True)
        self.dec_act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

        for channels in (256, 128, 64, 32, 16):
            self.batch_norms.append(nn.BatchNorm2d(channels))
        self.batch_norms.append(nn.BatchNorm2d(1))

        for index in range(6):
            in_channels = 1 if index == 0 else 2**index * 8
            out_channels = 2**index * 16
            self.encoder.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
            )

        for index in range(5, -1, -1):
            in_channels = 2**index * (32 if index != 5 else 16)
            out_channels = 1 if index == 0 else 2**index * 8
            self.decoder.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                )
            )

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        skips = []
        x = mix
        for layer in self.encoder:
            x = self.enc_act(layer(x))
            skips.append(x)
        skips.pop()

        for index, layer in enumerate(self.decoder):
            x = layer(x)
            x = self.batch_norms[index](x)
            if index < len(self.decoder) - 1:
                x = self.dec_act(x)
                if index < 3:
                    x = self.dropout(x)
                x = torch.cat([x, skips.pop()], dim=1)
            else:
                x = self.sigmoid(x)
        return x * mix

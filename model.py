import torch.nn as nn
import torch


class UNET(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        # Activation functions
        self.activation_encoder = nn.LeakyReLU(negative_slope=0.2)
        self.activation_decoder = nn.ReLU()
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        # Sigmoid activation for final output
        self.sigmoid = nn.Sigmoid()

        # Adding batch normalization layers for encoding
        for i in range(5, 0, -1):
            self.batch_norms.append(nn.BatchNorm2d(2**i * 8))
        self.batch_norms.append(nn.BatchNorm2d(1))

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Creating encoder layers
        for i in range(6):
            self.encoder_layers.append(
                nn.Conv2d(
                    2**i * 8 if i > 0 else 1,
                    2**i * 16,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                )
            )

        # Creating decoder layers
        for i in range(5, -1, -1):
            self.decoder_layers.append(
                nn.ConvTranspose2d(
                    2**i * (32 if i != 5 else 16),
                    2**i * 8 if i > 0 else 1,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                )
            )

    def forward(self, input):
        x = input.clone()
        encoder_outputs = []

        # Forward pass through encoder layers
        for layer in self.encoder_layers:
            x = self.activation_encoder(layer(x))
            encoder_outputs.append(x)

        encoder_outputs.pop()

        # Forward pass through decoder layers
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            x = self.batch_norms[i](x)
            x = (
                self.activation_decoder(x)
                if i < len(self.decoder_layers) - 1
                else self.sigmoid(x)
            )

            # Applying dropout for the first three decoder layers
            if i < 3:
                x = self.dropout(x)

            # Concatenating with the corresponding encoder output
            if len(encoder_outputs) > 0:
                x = torch.cat([x, encoder_outputs.pop()], dim=1)

        return x * input  # Element-wise multiplication with the input

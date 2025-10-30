# src/model.py
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """
    Input shape: (batch, seq_len, features)
    We encode → bottleneck → decode
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
      # x: (B, T, F)
      enc_out, (h, c) = self.encoder(x)      # enc_out: (B, T, H)
      # take last timestep
      last_h = enc_out[:, -1, :]             # (B, H)
      z = self.to_latent(last_h)             # (B, latent)

      # repeat latent across timesteps
      B, T, F = x.shape
      dec_in = self.decoder_input(z)         # (B, H)
      dec_in = dec_in.unsqueeze(1).repeat(1, T, 1)  # (B, T, H)

      dec_out, _ = self.decoder(dec_in)      # (B, T, H)
      out = self.output_layer(dec_out)       # (B, T, F)
      return out, z

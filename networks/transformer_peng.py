import torch
import torch.nn as nn

import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size=88, conv_out_channels=64, conv_kernel_size=3,
                 hidden_size=512, num_layers=4, out_size=6, nhead=8):
        super(LSTM, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_size,
                               out_channels=conv_out_channels,
                               kernel_size=conv_kernel_size,
                               padding=conv_kernel_size // 2)

        # Optional positional encoding for Transformer
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, conv_out_channels))  # (1, max_seq_len, emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_out_channels,
            nhead=nhead,
            dim_feedforward=hidden_size,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(conv_out_channels, out_size)

    def forward(self, x, y=None):
        # x: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # (batch, input_size, seq_len)
        x = self.conv1(x)       # (batch, conv_out_channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, conv_out_channels)

        # Add positional encoding (optional but often useful)
        if x.size(1) <= self.pos_embedding.size(1):
            x = x + self.pos_embedding[:, :x.size(1), :]
        else:
            raise ValueError("Input sequence length exceeds max positional embedding length")

        x = self.transformer(x)  # (batch, seq_len, conv_out_channels)
        x = self.fc(x)           # (batch, seq_len, out_size)
        return x
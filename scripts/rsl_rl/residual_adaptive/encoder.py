import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self,num_encoder_obs=21, d_model=128, nhead=8, num_layers=2, output_dim=32, num_time_steps=5):
        super().__init__()
        self.d_model = d_model
        
        # input projection
        self.input_projection = nn.Linear(num_encoder_obs, d_model)
        
        # time step position encoding
        self.positional_encoding = nn.Parameter(torch.randn(num_time_steps, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x: (batch_size, 5, 21)
        x = self.input_projection(x)  # (batch_size, 5, d_model)
        x = x + self.positional_encoding.unsqueeze(0) 
        x = self.transformer(x)  # (batch_size, 5, d_model)
        x = x.mean(dim=1)  # Global average pooling: (batch_size, d_model)
        x = self.output_projection(x)  # (batch_size, output_dim)
        return x
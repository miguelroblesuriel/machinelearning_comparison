import torch
import torch.nn as nn
import numpy as np

class FourierFeatures(nn.Module):
    def __init__(self, input_dim, num_features):
        super(FourierFeatures, self).__init__()
        self.B = nn.Parameter(torch.randn(input_dim, num_features) * 2 * np.pi, requires_grad=False)

    def forward(self, x):
        projection = torch.matmul(x, self.B)  # Matrix multiplication
        return torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1)

class Bertabolomics(nn.Module):
    def __init__(self, input_dim=2, num_features=64, d_model=128, num_heads=8, num_layers=6,
                 dim_feedforward=256, dropout=0.1):
        super(Bertabolomics, self).__init__()
        self.fourier_features = FourierFeatures(input_dim, num_features)
        self.input_projection = nn.Linear(2 * num_features, d_model)  # Project Fourier features to d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, attention_mask=None):
        x = self.fourier_features(x)
        x = self.input_projection(x)

        # TODO:
        # Create a mask for the transformer if an attention mask is provided
        if attention_mask is not None:
            attention_mask = attention_mask == 0  # Convert attention mask to appropriate format
        
        # Apply transformer encoder
        x = self.transformer_encoder(x.transpose(0, 1), src_key_padding_mask=attention_mask)
        x = x.transpose(0, 1)
        
        return x

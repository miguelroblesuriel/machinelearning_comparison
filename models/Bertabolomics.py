import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam

def index_with_mask(output, mask):
    return output[mask.unsqueeze(-1).expand_as(output)]


class MLP(nn.Module):
    def __init__(self, input_size=128, hidden_size=1024, output_size=1, dropout_prob=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None, padding_mask=None):
        x = self.fourier_features(x)
        x = self.input_projection(x)
        x = self.transformer_encoder(x, mask=mask, src_key_padding_mask=padding_mask)
        return x

def compute_triplet_loss(output, spectra):
    raise NotImplementedError("This function is not implemented yet.")


class BertabolomicsLightning(pl.LightningModule):
    def __init__(self, model, proj_model, learning_rate=1e-4, mode='pretrain'):
        super(BertabolomicsLightning, self).__init__()
        self.model = model
        self.proj_model = proj_model
        self.learning_rate = learning_rate
        self.mode = mode

    def forward(self, x, mask=None, padding_mask=None):
        return self.model(x, mask, padding_mask)

    def training_step(self, batch, batch_idx):
        (spectra, masked_spectra, 
            net_output_masks, spectra_masks, padding_masks) = batch
        output = self(masked_spectra, padding_mask=padding_masks)
        # Compute the loss based on the mode
        if self.mode == 'pretrain':
            # FIXME: m/z are not masked and this evaluates how good the m/z proj was. This should be fixed in the 
            # data loaders?
            proj_output = self.proj_model(output[net_output_masks])
            loss = F.mse_loss(proj_output, spectra[spectra_masks].unsqueeze(-1))
        elif self.mode == 'triplet':
            # Compute triplet loss
            loss = compute_triplet_loss(output, spectra)
        else:
            raise ValueError("Unsupported mode. Please choose either 'pretrain' or 'triplet'.")
        
        # Log the training loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (spectra, masked_spectra, 
            net_output_masks, spectra_masks, padding_masks) = batch
        output = self(masked_spectra, padding_mask=padding_masks)
        
        # Compute the loss based on the mode
        if self.mode == 'pretrain':
            proj_output = self.proj_model(output[net_output_masks])
            val_loss = F.mse_loss(proj_output, spectra[spectra_masks].unsqueeze(-1))
        elif self.mode == 'triplet':
            # Compute triplet loss
            val_loss = compute_triplet_loss(output, spectra)
        else:
            raise ValueError("Unsupported mode. Please choose either 'pretrain' or 'triplet'.")
        
        # Log the validation loss
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
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
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None, padding_mask=None):
        x = self.fourier_features(x)
        print(x[:5, ...])
        x = self.input_projection(x)
        x = self.transformer_encoder(x, mask=mask, src_key_padding_mask=padding_mask)
        return x

class BertabolomicsLightning(pl.LightningModule):
    def __init__(self, model, proj_model, learning_rate=1e-4, mode='pretrain'):
        super(BertabolomicsLightning, self).__init__()
        self.model = model
        self.proj_model = proj_model
        self.learning_rate = learning_rate
        self.mode = mode
        # Careful, if using cosine we should select the margin carefully
        # self.triplet_loss = (
        #     nn.TripletMarginWithDistanceLoss(
        #         distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
        # )
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, x, mask=None, padding_mask=None):
        return self.model(x, mask, padding_mask)

    def training_step(self, batch, batch_idx):
        # Compute the loss based on the mode
        if self.mode == 'pretrain':
            raise NotImplementedError("Pretrain mode is not implemented yet.")
            # Following lines are from previous version and should be reviewed 
            # to recall the expected flow
            # (spectra, masked_spectra, 
            #     net_output_masks, spectra_masks, padding_masks) = batch
            # output = self(masked_spectra, padding_mask=padding_masks)
            #
            # # FIXME: m/z are not masked and this evaluates how good the m/z proj was. This should be fixed in the 
            # # data loaders?
            # proj_output = self.proj_model(output[net_output_masks])
            # loss = F.mse_loss(proj_output, spectra[spectra_masks].unsqueeze(-1))
        elif self.mode == 'triplet':
            # TODO: execute line by line
            # Compute triplet loss
            (anchors, positives, negatives, anchor_padding_masks, positive_padding_masks, negative_padding_masks) = batch
            anchors_output = self(anchors, padding_mask=anchor_padding_masks)
            positives_output = self(positives, padding_mask=positive_padding_masks) 
            negatives_output = self(negatives, padding_mask=negative_padding_masks)
            # Each of the outputs has shape (batch_size, seq_len, d_model), but
            # we need to pick the CLS token for spectra comparison, which is on the
            # first position of seq_len
            anchors_output = anchors_output[:, 0, :]
            positives_output = positives_output[:, 0, :]
            negatives_output = negatives_output[:, 0, :]
            loss = self.triplet_loss(anchors_output, positives_output, negatives_output) 
        else:
            raise ValueError("Unsupported mode. Please choose either 'pretrain' or 'triplet'.")
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.logger.experiment.add_scalars('loss', {'train': loss}, self.global_step) 
        return loss

    def validation_step(self, batch, batch_idx):
        if self.mode == 'pretrain':
            raise NotImplementedError("Pretrain mode is not implemented yet.")
        elif self.mode == 'triplet':
            (anchors, positives, negatives, anchor_padding_masks, positive_padding_masks, negative_padding_masks) = batch
            anchors_output = self(anchors, padding_mask=anchor_padding_masks)
            positives_output = self(positives, padding_mask=positive_padding_masks) 
            negatives_output = self(negatives, padding_mask=negative_padding_masks)
            # Pick the CLS token for spectra comparison
            anchors_output = anchors_output[:, 0, :]
            positives_output = positives_output[:, 0, :]
            negatives_output = negatives_output[:, 0, :]
            loss = self.triplet_loss(anchors_output, positives_output, negatives_output) 
        else:
            raise ValueError("Unsupported mode. Please choose either 'pretrain' or 'triplet'.")
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.logger.experiment.add_scalars('loss', {'val': loss}, self.global_step) 
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

import hashlib
import pickle
from re import S

import numpy as np
import pytorch_lightning as pl
from massql import msql_fileloading
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from machinelearning_model.CustomSpectraDataset import (CustomSpectraDataset,
                                                        collate_fn,
                                                        load_triplets_dataset)
from models.Bertabolomics import MLP, Bertabolomics, BertabolomicsLightning


def hash_array(x):
    hasher = hashlib.sha256()
    hasher.update(x.tobytes())
    return hasher.digest()


if __name__ == "__main__":
    mzml_filename = "datafiles/049_Blk_Water_NEG.mzMl"
    npy_triplet_filename = 'datafiles/049_Blk_Water_NEG_triplets.npy'
    # TODO: scaler
    dataset = load_triplets_dataset(mzml_filename, npy_triplet_filename)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)
    # dataloader_pretraining = DataLoader(
    #     dataset, batch_size=8, collate_fn=collate_fn_pretraining
    # )
    print("Loading done!")

    base_model = Bertabolomics()
    proj_head = MLP() # unused for triplet training
    model = BertabolomicsLightning(base_model, proj_head, mode="triplet")

    trainer_pretrain = pl.Trainer(max_epochs=2, accelerator="cuda")
    print("Training...")
    trainer_pretrain.fit(model, dataloader)


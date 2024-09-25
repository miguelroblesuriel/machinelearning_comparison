from re import S
from massql import msql_fileloading
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from data.MaskedMS2 import MS2Dataset, collate_fn_pretraining
from data.utils import load_spectra
from machinelearning_model.CustomSpectraDataset import CustomSpectraDataset, collate_fn
import pickle
from sklearn.preprocessing import StandardScaler

from models.Bertabolomics import BertabolomicsLightning, Bertabolomics, MLP


import hashlib


def hash_array(x):
    hasher = hashlib.sha256()
    hasher.update(x.tobytes())
    return hasher.digest()


if __name__ == "__main__":
    print("Loading...")
    spectra_filename = "spectra.pkl"
    try:
        with open(spectra_filename, "rb") as f:
            spectra = pickle.load(f)
    except FileNotFoundError:
        spectra = []
        input_filename = "/data/tino/triplet_loss/049_Blk_Water_NEG.mzMl"
        ms1_df, ms2_df = msql_fileloading.load_data(input_filename, cache="feather")

        for idx in ms2_df["scan"].unique():
            spectra.append(load_spectra(ms2_df, idx))

        with open(spectra_filename, "wb") as f:
            pickle.dump(spectra, f)

    normalizer = StandardScaler() 
    dataset = MS2Dataset(spectra, normalizer)

    dataloader_pretraining = DataLoader(
        dataset, batch_size=8, collate_fn=collate_fn_pretraining
    )
    print("Loading done!")

    base_model = Bertabolomics()
    proj_head = MLP()
    model = BertabolomicsLightning(base_model, proj_head, mode="pretrain")

    trainer_pretrain = pl.Trainer(max_epochs=2, accelerator="cpu")
    print("Training...")
    trainer_pretrain.fit(model, dataloader_pretraining)

# Example usage with DataLoader
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from massql import msql_fileloading
from torch.utils.data import Dataset
from preprocessing.peak_processing import peak_processing
from comparison.createSpectrum import createSpectrum

def collate_fn(batch, max_len=174):
    # batch is a list of tuples of 3 elements. Each element is a spectrum which should be collated 
    # with collate_spectrum
    anchors = []
    positives = []
    negatives = []
    anchor_padding_masks = []
    positive_padding_masks = []
    negative_padding_masks = []
    for triplet in batch:
        anchor_spectrum, anchor_padding_mask = collate_spectrum(triplet[0], max_len)
        positive_spectrum, positive_padding_mask = collate_spectrum(triplet[1], max_len)
        negative_spectrum, negative_padding_mask = collate_spectrum(triplet[2], max_len)
        anchors.append(anchor_spectrum)
        positives.append(positive_spectrum)
        negatives.append(negative_spectrum)
        anchor_padding_masks.append(anchor_padding_mask)
        positive_padding_masks.append(positive_padding_mask)
        negative_padding_masks.append(negative_padding_mask)

    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives),
        torch.stack(anchor_padding_masks),
        torch.stack(positive_padding_masks),
        torch.stack(negative_padding_masks)
    )

def collate_spectrum(spectrum, max_len, cls_token=(0.0, 0.0)):
    """max_len is the maximum length of the spectrum after padding and the inclusion of the cls_token"""
    spectrum_with_cls = np.vstack([np.array(cls_token), spectrum])
    length = len(spectrum_with_cls)
    # Pad the spectrum
    padded_spectrum = np.pad(spectrum_with_cls, ((0, max_len - length), (0, 0)), mode='constant', constant_values=0)
    # In torch transformer, (https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
    # if a boolean tensor is provided for any of the [src/tgt/memory]_mask arguments, positions with a True value are not allowed to participate in the attention
    padding_mask = [False] * length + [True] * (max_len - length)

    padded_spectra = torch.tensor(padded_spectrum, dtype=torch.float32)
    padding_masks = torch.tensor(padding_mask, dtype=torch.int8)
    
    return padded_spectra, padding_masks

class CustomSpectraDataset(Dataset):
    def __init__(self, dupla, triplets, comparison_scores, ms2_df):
        self.duplas = dupla
        self.triplets = triplets
        self.comparison_scores = comparison_scores
        self.ms2_df = ms2_df

    def _load_spectra(self, scan_idx):
        return self.ms2_df[self.ms2_df['scan'] == scan_idx][['mz', 'i_norm']].to_numpy()

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx, preprocessing = False):
        dupla = self.duplas[idx]
        triplet = self.triplets[idx]
        scores = self.comparison_scores[idx]
        probabilities = [ (i ** 3) for i in scores]
        total_probability = sum(probabilities)
        probabilities = [prob / total_probability for prob in probabilities]
        random_index = np.random.choice(range(len(triplet)), p=probabilities)
        random_triplet = triplet[random_index]
        return (
            self._load_spectra(dupla[0]),
            self._load_spectra(dupla[1]),
            self._load_spectra(random_triplet)
        )


if __name__ == "__main__":
    input_filename = "/data/tino/triplet_loss/049_Blk_Water_NEG.mzMl"
    ms1_df, ms2_df = msql_fileloading.load_data(input_filename, cache='feather') 

    file_path = '/data/tino/triplet_loss/049_Blk_Water_NEG_triplets.npy'
    loaded_data = np.load(file_path, allow_pickle=True)
    duplas = []
    triplets = []
    scores = []
    for item in loaded_data:
        if item["triplet"] != []:
            duplas.append(item["dupla"].tolist())
            triplets.append(item["triplet"])
            scores.append(item["scores"])

    dataset = CustomSpectraDataset(duplas, triplets, scores, ms2_df)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    print(next(iter(dataloader)))

    # max_peaks = 0
    # all_peaks = []
    # for a, b, c in dataset:
    #     all_peaks.append((a.shape[0], b.shape[0], c.shape[0]))
    # print(a.shape)
    # print(b.shape)
    # print(c.shape)
    # # save all peaks in csv using numpy
    # import numpy as np
    # np.savetxt("all_peaks.csv", np.stack(all_peaks), delimiter=",")
    #
    # print(f"Max peaks: {max(all_peaks)}")
    # # 173, 96, 140
    # 
    #

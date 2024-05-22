from massql import msql_fileloading
import numpy as np
from torch.utils.data import Dataset, DataLoader
from machinelearning_model.CustomSpectraDataset import CustomSpectraDataset, collate_fn

from models.FourierFeatures import FourierFeatures


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
    
    a, p, n, am, pm, nm = next(iter(dataloader))
    print(a.shape)
    fourier = FourierFeatures(2, 3)
    print(fourier(a).shape)

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
 

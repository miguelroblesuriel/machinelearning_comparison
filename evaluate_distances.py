import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preprocessing.convert_to_spectra import convert_to_spectra

from comparison.cosine_greedy import cosine_greedy

from data.CustomSpectraDataset import collate_fn, load_triplets_dataset
from models.Bertabolomics import MLP, Bertabolomics, BertabolomicsLightning

if __name__ == "__main__":
    dataset_filename = "datafiles/049_Blk_Water_NEG"
    # TODO: automatically get last checkpoint
    checkpoint_path = "lightning_logs/bertabolomics.ckpt"

    # TODO: Reproduce changes in main.py and avoid code dup
    train_dataset, test_dataset = load_triplets_dataset(dataset_filename)
    dataloader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn)
    # dataloader_pretraining = DataLoader(
    #     dataset, batch_size=8, collate_fn=collate_fn_pretraining
    # )
    print("Loading done!")

    base_model = Bertabolomics()
    proj_head = MLP() # unused for triplet training
    model = BertabolomicsLightning.load_from_checkpoint(
        checkpoint_path, 
        model=base_model, proj_model=proj_head, mode="triplet"
    )
    def compute_distance(spectra1, spectra1_padding_mask,
                         spectra2, spectra2_padding_mask):
        with torch.no_grad():
            model.eval()
            device = model.device
            # TODO: transform the next lines into a method of BertabolomicsLightning
            embedding1 = model(spectra1.to(device), padding_mask=spectra1_padding_mask.to(device))

            embedding2 = model(spectra2.to(device), padding_mask=spectra2_padding_mask.to(device))
            embedding1_cls = embedding1[:, 0, :]
            embedding2_cls = embedding2[:, 0, :]
        return F.pairwise_distance(embedding1_cls, embedding2_cls)
    

    # Get some spectra for an example of computing distances
    (anchors, positives, negatives, anchor_padding_masks, positive_padding_masks, negative_padding_masks) = next(iter(dataloader))
    anchors_spectra = convert_to_spectra(anchors, anchor_padding_masks)
    positives_spectra = convert_to_spectra(positives, positive_padding_masks)
    negatives_spectra = convert_to_spectra(negatives, negative_padding_masks)

    scores = []
    for anchor, positive in zip(anchors_spectra, positives_spectra):
        spectra = []
        spectra.append(anchor)
        spectra.append(positive)
        scores.append(cosine_greedy(0.005, spectra, spectra))
    final_scores = []
    for score in scores:
        print(score.scores[0][1]["score"])
        final_scores.append(score.scores[0][1]["score"])
    distances = compute_distance(anchors, anchor_padding_masks, positives, positive_padding_masks)
    print(distances)



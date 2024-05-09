import os
from random import random
import numpy
import pandas as pd
from torch.utils.data import Dataset
from preprocessing.peak_processing import peak_processing
from comparison.createSpectrum import createSpectrum

class CustomSpectraDataset(Dataset):
    def __init__(self, dupla, triplets, comparison_scores):
        self.duplas = dupla
        self.triplets = triplets
        self.comparison_scores = comparison_scores

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx, preprocessing = False):
        dupla = self.duplas[idx]
        triplet = self.triplets[idx]
        scores = self.comparison_scores[idx]
        probabilities = [ (i ** 3) for i in scores]
        total_probability = sum(probabilities)
        probabilities = [prob / total_probability for prob in probabilities]
        random_index = numpy.random.choice(range(len(triplet)), p=probabilities)

        random_triplet = triplet[random_index]
        anchor = createSpectrum(dupla[0])
        reference = createSpectrum(dupla[1])
        loss = createSpectrum(random_triplet)
        if preprocessing:
            anchor = peak_processing(anchor)
            reference = peak_processing(reference)
            loss = peak_processing(loss)
        return anchor,reference, loss
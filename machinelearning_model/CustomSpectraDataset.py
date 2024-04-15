import os
from random import random
import numpy
import pandas as pd
from torch.utils.data import Dataset

class CustomSpectraDataset(Dataset):
    def __init__(self, dupla, triplets, comparison_scores):
        self.duplas = dupla
        self.triplets = triplets
        self.comparison_scores = comparison_scores

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        dupla = self.duplas[idx]
        triplet = self.triplets[idx]
        scores = self.comparison_scores[idx]
        probabilities = [ (i ** 3) for i in scores]
        total_probability = sum(probabilities)
        probabilities = [prob / total_probability for prob in probabilities]
        random_index = numpy.random.choice(range(len(triplet)), p=probabilities)

        random_triplet = triplet[random_index]

        return dupla[0],dupla[1], random_triplet
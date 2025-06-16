
from matchms import calculate_scores
from matchms.similarity import CosineGreedy
from visualization.mirror_plot import mirror_plot
import numpy as np

def cosine_greedy(tolerance, spectrums1, spectrums2):
    similarity_measure = CosineGreedy(tolerance=tolerance)
    spectrums1 = np.array(spectrums1)
    spectrums2 = np.array(spectrums2)
    scores = calculate_scores(spectrums1, spectrums2, similarity_measure, is_symmetric=False)
    return scores
from ms2deepscore import MS2DeepScore
from matchms import calculate_scores

def ms2deepscore(model, spectrums1, spectrums2=None):
    similarity_measure = MS2DeepScore(model)
    if spectrums2 == None:
        scores = calculate_scores(spectrums1, spectrums1, similarity_measure, is_symmetric=True)
        return scores
    else:
        scores = calculate_scores(spectrums1, spectrums2, similarity_measure, is_symmetric=False)
        return scores
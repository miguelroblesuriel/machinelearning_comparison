from matchms import calculate_scores
from spec2vec import Spec2Vec

def spec2vec(model, spectrums1, spectrums2=None):
    spec2vec_similarity = Spec2Vec(model=model, intensity_weighting_power=0.5, allowed_missing_percentage=5.0)
    if spectrums2 == None:
        scores = calculate_scores(spectrums1, spectrums1, spec2vec_similarity, is_symmetric=True)
        return scores
    else:
        scores = calculate_scores(spectrums1, spectrums2, spec2vec_similarity, is_symmetric=False)
        return scores


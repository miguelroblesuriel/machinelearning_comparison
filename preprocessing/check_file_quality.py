from comparison.cosine_greedy import cosine_greedy
from comparison.createSpectrum import createSpectrum
import numpy
import matplotlib.pyplot as plt
def check_file_quality(scans,ms2_df):
    spectra = []
    for scan in scans:
        spectra.append(createSpectrum(ms2_df[ms2_df['scan'] == scan]['i_norm'].to_numpy(),
                                      numpy.sort(ms2_df[ms2_df['scan'] == scan]['mz'].to_numpy()),
                                      ms2_df[ms2_df['scan'] == scan]['precmz'].unique()))
    scores = cosine_greedy(0.005, spectra, spectra)
    scores_array = scores.scores.to_array()
    oneD_scores = scores_array["CosineGreedy_score"].flatten()
    print(oneD_scores)
    plt.hist(oneD_scores, bins=50, range=(0, 1), edgecolor='black', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Matrix Values')
    plt.show()
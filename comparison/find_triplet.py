import numpy
from comparison.cosine_greedy import cosine_greedy
from comparison.createSpectrum import createSpectrum

def find_triplet(dupla, features_scans, ms2_df):

    threshold = 0.75
    peak_threshold = 10
    scan_dupla = dupla.iloc[0]
    triplet_scan = []
    spectrum_dupla = []
    spectrum_dupla.append(createSpectrum(ms2_df[ms2_df['scan'] == scan_dupla]['i_norm'].to_numpy(),
                                         numpy.sort(ms2_df[ms2_df['scan'] == scan_dupla]['mz'].to_numpy()), ms2_df[ms2_df['scan'] == scan_dupla]['precmz'].unique()))
    spectrum_dupla.append(createSpectrum(ms2_df[ms2_df['scan'] == scan_dupla]['i_norm'].to_numpy(),
                                         numpy.sort(ms2_df[ms2_df['scan'] == scan_dupla]['mz'].to_numpy()), ms2_df[ms2_df['scan'] == scan_dupla]['precmz'].unique()))
    comparison_scores=[]
    for f in features_scans:
        spectra = []
        if scan_dupla not in f:
            for scan in f:
                spectra.append(createSpectrum(ms2_df[ms2_df['scan'] == scan]['i_norm'].to_numpy(),
                                              numpy.sort(ms2_df[ms2_df['scan'] == scan]['mz'].to_numpy()),ms2_df[ms2_df['scan'] == scan]['precmz'].unique()))
                spectra.append(createSpectrum(ms2_df[ms2_df['scan'] == scan]['i_norm'].to_numpy(),
                                              numpy.sort(ms2_df[ms2_df['scan'] == scan]['mz'].to_numpy()), ms2_df[ms2_df['scan'] == scan]['precmz'].unique()))
                scores = cosine_greedy(0.005, spectra, spectrum_dupla)
                scores_array = scores.scores
                if scores_array["score"][0][0] > threshold:
                    if scores_array["matches"][0][0] > peak_threshold:
                        triplet_scan.append(scan)
                        comparison_scores.append(scores_array["score"][0][0])
                spectra.clear()
    return triplet_scan, comparison_scores
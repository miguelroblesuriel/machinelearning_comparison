import numpy
from comparison.cosine_greedy import cosine_greedy
from comparison.createSpectrum import createSpectrum

def find_triplet(dupla, features_scans, ms2_df):

    threshold = 0.75
    scan_dupla = dupla.iloc[0]
    triplet_scan = []
    spectrum_dupla = []
    spectrum_dupla.append(createSpectrum(ms2_df[ms2_df['scan'] == scan_dupla]['i_norm'].to_numpy(),
                                         numpy.sort(ms2_df[ms2_df['scan'] == scan_dupla]['mz'].to_numpy())))
    spectrum_dupla.append(createSpectrum(ms2_df[ms2_df['scan'] == scan_dupla]['i_norm'].to_numpy(),
                                         numpy.sort(ms2_df[ms2_df['scan'] == scan_dupla]['mz'].to_numpy())))
    for f in features_scans:
        spectra = []
        if scan_dupla not in f:
            for scan in f:
                spectra.append(createSpectrum(ms2_df[ms2_df['scan'] == scan]['i_norm'].to_numpy(),
                                              numpy.sort(ms2_df[ms2_df['scan'] == scan]['mz'].to_numpy())))
                spectra.append(createSpectrum(ms2_df[ms2_df['scan'] == scan]['i_norm'].to_numpy(),
                                              numpy.sort(ms2_df[ms2_df['scan'] == scan]['mz'].to_numpy())))
                scores = cosine_greedy(0.005, spectra, spectrum_dupla)
                scores_array = scores.scores
                if scores_array["score"][0][0] > threshold:
                    triplet_scan.append(scan)
                spectra.clear()
    return triplet_scan
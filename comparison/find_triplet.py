import numpy
from comparison.cosine_greedy import cosine_greedy
from comparison.createSpectrum import createSpectrum
from preprocessing.peak_processing import peak_processing
from visualization.mirror_plot import mirror_plot


def find_triplet(dupla, features_scans, features_scans_precmz, ms2_df,threshold,peak_threshold):

    scan_dupla = dupla.iloc[0]
    triplet_scan = []
    spectrum_dupla = []
    filtered_spectrum_dupla = []
    spectrum_dupla.append(createSpectrum(ms2_df[ms2_df['scan'] == scan_dupla]['i_norm'].to_numpy(),
                                         (numpy.sort(ms2_df[ms2_df['scan'] == scan_dupla]['mz'].to_numpy())), ms2_df[ms2_df['scan'] == scan_dupla]['precmz'].unique(), scan_dupla))
    spectrum_dupla.append(createSpectrum(ms2_df[ms2_df['scan'] == scan_dupla]['i_norm'].to_numpy(),
                                         (numpy.sort(ms2_df[ms2_df['scan'] == scan_dupla]['mz'].to_numpy())), ms2_df[ms2_df['scan'] == scan_dupla]['precmz'].unique(), scan_dupla))
    for s in spectrum_dupla:
        filtered_spectrum_dupla.append(peak_processing(s))
    comparison_scores=[]
    for f, precmz in zip(features_scans, features_scans_precmz):
        spectra = []
        filtered_spectra = []
        if scan_dupla not in f or precmz !=ms2_df[ms2_df['scan'] == scan_dupla]['precmz'].unique():
            for scan in f:
                spectra.append(createSpectrum(ms2_df[ms2_df['scan'] == scan]['i_norm'].to_numpy(),
                                              numpy.sort(ms2_df[ms2_df['scan'] == scan]['mz'].to_numpy()),ms2_df[ms2_df['scan'] == scan]['precmz'].unique(), scan))
                spectra.append(createSpectrum(ms2_df[ms2_df['scan'] == scan]['i_norm'].to_numpy(),
                                              numpy.sort(ms2_df[ms2_df['scan'] == scan]['mz'].to_numpy()),
                                              ms2_df[ms2_df['scan'] == scan]['precmz'].unique(), scan))
                for s in spectra:
                    filtered_spectra.append(peak_processing(s))
                scores = cosine_greedy(0.005, filtered_spectra, filtered_spectrum_dupla)
                scores_array = scores.scores.to_array()
                if scores_array["CosineGreedy_score"][0][0] > threshold:
                    if scores_array["CosineGreedy_matches"][0][0] > peak_threshold:
                        triplet_scan.append(scan)
                        comparison_scores.append(scores_array["CosineGreedy_score"][0][0])
                spectra.clear()
                filtered_spectra.clear()
    return triplet_scan, comparison_scores
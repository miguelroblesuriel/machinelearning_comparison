from comparison.cosine_greedy import cosine_greedy
import numpy

def compare_two_scans(scan1, scan2, ms2_df):
    spectra = []
    spectra.append(createSpectrum(ms2_df[ms2_df['scan'] == scan1]['i'].to_numpy(),
                                  numpy.sort(ms2_df[ms2_df['scan'] == scan1]['mz'].to_numpy())))
    spectra.append(createSpectrum(ms2_df[ms2_df['scan'] == scan2]['i'].to_numpy(),
                                  numpy.sort(ms2_df[ms2_df['scan'] == scan2]['mz'].to_numpy())))
    scores = cosine_greedy(0.005, spectra, spectra)
    return scores
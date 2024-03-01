import json
import os
import uuid
import pandas as pd
import numpy as np

from tqdm import tqdm
from matchms.importing import load_from_mgf
import pymzml
from pyteomics import mzxml, mzml

import logging

from matchms import calculate_scores
from matchms.similarity import CosineGreedy
from matchms import Spectrum

from matplotlib import pyplot as plt

from pyopenms import *

from massql import msql_fileloading

import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus


def mirror_plot(spectrum1,spectrum2,i,precmz1,precmz2,rt1,rt2):
    plt.figure(figsize=(8, 4))
    # Plot Spectrum 1
    plt.stem(spectrum1.mz, spectrum1.intensities, linefmt='b-', markerfmt='', basefmt=' ', label=f'Spectrum 1- precmz={precmz1} - rt={rt1}' )

    # Plot Spectrum 2 with reversed m/z values
    plt.stem(spectrum2.mz, -spectrum2.intensities, linefmt='r-', markerfmt='', basefmt=' ',
             label=f'Spectrum 2 (Mirrored) - precmz={precmz2} - rt={rt2}' )
    # Add labels and legend
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.title('Mirror Plot Along X-axis of MS2 Spectra')
    plt.legend()

    # Show the mirror plot
    plt.savefig("mirror " + str(i)+ ".png", dpi=300, bbox_inches="tight", transparent=True)
def cosine_greedy(tolerance, spectrums1, spectrums2):
    similarity_measure = CosineGreedy(tolerance=tolerance)
    scores = calculate_scores(spectrums1, spectrums2, similarity_measure, is_symmetric=False)
    return scores

def compare_two_scans(scan1,scan2):
    spectra = []
    spectra.append(createSpectrum(ms2_df[ms2_df['scan'] == scan1]['i'].to_numpy(),
                                  numpy.sort(ms2_df[ms2_df['scan'] == scan1]['mz'].to_numpy())))
    spectra.append(createSpectrum(ms2_df[ms2_df['scan'] == scan2]['i'].to_numpy(),
                                  numpy.sort(ms2_df[ms2_df['scan'] == scan2]['mz'].to_numpy())))
    scores = cosine_greedy(0.005, spectra, spectra)

    scores_array = scores.scores
    plt.figure(figsize=(6, 6), dpi=150)
    plt.imshow(scores_array["score"], cmap="viridis")
    plt.colorbar(shrink=0.7)
    plt.title("Cosine Greedy spectra similarities")
    plt.xlabel("Spectrum #ID")
    plt.ylabel("Spectrum #ID")
    plt.show()

def find_triplet(dupla,features_scans,ms2_df):
    threshold=0.75
    scan_dupla = dupla.iloc[0]
    triplet_scan = []
    spectrum_dupla = []
    spectrum_dupla.append(createSpectrum(ms2_df[ms2_df['scan'] == scan_dupla]['i_norm'].to_numpy(),numpy.sort(ms2_df[ms2_df['scan'] == scan_dupla]['mz'].to_numpy())))
    spectrum_dupla.append(createSpectrum(ms2_df[ms2_df['scan'] == scan_dupla]['i_norm'].to_numpy(),numpy.sort(ms2_df[ms2_df['scan'] == scan_dupla]['mz'].to_numpy())))
    for f in features_scans:
        spectra=[]
        if scan_dupla not in f:
            for scan in f:
                spectra.append(createSpectrum(ms2_df[ms2_df['scan'] == scan]['i_norm'].to_numpy(),numpy.sort(ms2_df[ms2_df['scan'] == scan]['mz'].to_numpy())))
                spectra.append(createSpectrum(ms2_df[ms2_df['scan'] == scan]['i_norm'].to_numpy(),numpy.sort(ms2_df[ms2_df['scan'] == scan]['mz'].to_numpy())))
                scores = cosine_greedy(0.005,  spectra, spectrum_dupla)
                scores_array = scores.scores
                if scores_array["score"][0][0]> threshold:
                    triplet_scan.append(scan)
                spectra.clear()
    return triplet_scan

def _determine_scan_polarity_pyteomics_mzML(spec):
    """
    Gets an enum for positive and negative polarity, for pyteomics

    Args:
        spec ([type]): [description]

    Returns:
        [type]: [description]
    """
    polarity = 0

    if "negative scan" in spec:
        polarity = 2
    if "positive scan" in spec:
        polarity = 1

    return polarity

def createSpectrum(spectrum_i,spectrum_mz):
    spectrum=Spectrum(mz=spectrum_mz,
                       intensities=spectrum_i,
                       metadata= None,
                       metadata_harmonization= None)
    return spectrum

def maxMs2i(scan, ms2_df):
    maxMs2i = ms2_df.loc[ms2_df['scan']==scan]['i'].max
    return maxMs2i


input_filename = "049_Blk_Water_NEG.mzMl"
"""
previous_ms1_scan = 0

# MS1
all_mz = []
all_rt = []
all_polarity = []
all_i = []
all_i_norm = []
all_i_tic_norm = []
all_scan = []

# MS2
all_msn_mz = []
all_msn_rt = []
all_msn_polarity = []
all_msn_i = []
all_msn_i_norm = []
all_msn_i_tic_norm = []
all_msn_scan = []
all_msn_precmz = []
all_msn_ms1scan = []
all_msn_charge = []
all_msn_mobility = []

with mzml.read(input_filename) as reader:
    for spectrum in tqdm(reader):
        if len(spectrum["intensity array"]) == 0:
            continue

        # Getting the RT
        try:
            rt = spectrum["scanList"]["scan"][0]["scan start time"]
        except:
            rt = 0

        # Correcting the unit
        try:
            if spectrum["scanList"]["scan"][0]["scan start time"].unit_info == "second":
                rt = rt / 60
        except:
            pass

        scan = int(spectrum["id"].replace("scanId=", "").split("scan=")[-1])

        if not "m/z array" in spectrum:
            # This is not a mass spectrum
            continue

        mz = spectrum["m/z array"]
        intensity = spectrum["intensity array"]
        i_max = max(intensity)
        i_sum = sum(intensity)

        # If there is no ms level, its likely an UV/VIS spectrum and we can skip
        if not "ms level" in spectrum:
            continue

        mslevel = spectrum["ms level"]
        if mslevel == 1:
            all_mz += list(mz)
            all_i += list(intensity)
            all_i_norm += list(intensity / i_max)
            all_i_tic_norm += list(intensity / i_sum)
            all_rt += len(mz) * [rt]
            all_scan += len(mz) * [scan]
            all_polarity += len(mz) * [_determine_scan_polarity_pyteomics_mzML(spectrum)]
            previous_ms1_scan = scan

        if mslevel == 2:
            msn_mz = spectrum["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]["selected ion m/z"]
            msn_charge = 0

            if "charge state" in spectrum["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]:
                msn_charge = int(
                    spectrum["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]["charge state"])

            all_msn_mz += list(mz)
            all_msn_i += list(intensity)
            all_msn_i_norm += list(intensity / i_max)
            all_msn_i_tic_norm += list(intensity / i_sum)
            all_msn_rt += len(mz) * [rt]
            all_msn_scan += len(mz) * [scan]
            all_msn_polarity += len(mz) * [_determine_scan_polarity_pyteomics_mzML(spectrum)]
            all_msn_precmz += len(mz) * [msn_mz]
            all_msn_ms1scan += len(mz) * [previous_ms1_scan]
            all_msn_charge += len(mz) * [msn_charge]

            if "product ion mobility" in spectrum["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]:
                mobility = spectrum["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0][
                    "product ion mobility"]
                all_msn_mobility += len(mz) * [mobility]

ms1_df = pd.DataFrame()
if len(all_mz) > 0:
    ms1_df['i'] = all_i
    ms1_df['i_norm'] = all_i_norm
    ms1_df['i_tic_norm'] = all_i_tic_norm
    ms1_df['mz'] = all_mz
    ms1_df['scan'] = all_scan
    ms1_df['rt'] = all_rt
    ms1_df['polarity'] = all_polarity

ms2_df = pd.DataFrame()
if len(all_msn_mz) > 0:
    ms2_df['i'] = all_msn_i
    ms2_df['i_norm'] = all_msn_i_norm
    ms2_df['i_tic_norm'] = all_msn_i_tic_norm
    ms2_df['mz'] = all_msn_mz
    ms2_df['scan'] = all_msn_scan
    ms2_df['rt'] = all_msn_rt
    ms2_df["polarity"] = all_msn_polarity
    ms2_df["precmz"] = all_msn_precmz
    ms2_df["ms1scan"] = all_msn_ms1scan
    ms2_df["charge"] = all_msn_charge

    if len(all_msn_mobility) == len(all_msn_i):
        ms2_df["mobility"] = all_msn_mobility

all_i_filtered = []
for i in range(len(all_i_norm)):
    if all_i_norm[i] >= 0.2:
        all_i_filtered.append(i)

spectrum_i = []
spectrum_mz = []
actual_scan = all_scan[0]
spectra=[]
scans = ms1_df['scan'].unique()


# Prepare data loading (save memory by only
# loading MS1 spectra into memory)
options = PeakFileOptions()
options.setMSLevels([1])
fh = MzMLFile()
fh.setOptions(options)

# Load data
input_map = MSExperiment()
fh.load(input_filename, input_map)
input_map.updateRanges()

ff = FeatureFinder()
ff.setLogType(LogType.CMD)

# Run the feature finder
name = "centroided"
features = FeatureMap()
seeds = FeatureMap()
params = FeatureFinder().getParameters(name)
ff.run(name, input_map, features, params, seeds)

features.setUniqueIds()
fh = FeatureXMLFile()
fh.store("output.featureXML", features)
print("Found", features.size(), "features")
"""
ms1_df, ms2_df = msql_fileloading.load_data(input_filename, cache='feather')
featurexml_file = "C:/Users/migue/OneDrive/Escritorio/Uni/Beca/machinelearning_comparison/output.featureXML"

features = FeatureMap()

FeatureXMLFile().load(featurexml_file, features)

matched_scans=[]
spectra=[]
scans = ms2_df['scan'].unique()


"""
for f in features:
    print(ms1_df.loc[(
                (f.getRT() - 0.1 < ms1_df['rt']) & (ms1_df['rt'] < f.getRT() + 0.1))])
"""

i=0
duplas = []
maxMs2i=[]
features_scans=[]
for f in features:
    matched_scans=(ms2_df.loc[(
                    (f.getMZ() - 0.1 < ms2_df['precmz']) & (ms2_df['precmz'] < f.getMZ() + 0.1)) & (
                    (f.getRT()/60 - 0.1 < ms2_df['rt']) & (ms2_df['rt'] < f.getRT()/60 + 0.1))]['scan'].unique())
    features_scans.append(matched_scans)


    for m in matched_scans:
        maxMs2i.append(ms2_df.loc[ms2_df['scan']==m]['i'].max())
    Scan_intensity_dict = {
        'scan' : matched_scans,
        'maxMs2i' : maxMs2i
    }
    Scan_intensity_df=pd.DataFrame(Scan_intensity_dict)
    Scan_intensity_df_sorted = Scan_intensity_df.sort_values(by='maxMs2i')
    scans = Scan_intensity_df_sorted['scan']
    if len(scans) > 1:
        duplas.append(scans[-2:])
    """
    for scan in scans:
        spectra.append(createSpectrum(ms2_df[ms2_df['scan'] == scan]['i_norm'].to_numpy(),numpy.sort(ms2_df[ms2_df['scan'] == scan]['mz'].to_numpy())))
        
    mirror_plot(spectra[-1],spectra[-2],i)

    scores=cosine_greedy(0.005,spectra[-2:],spectra[-2:])

    scores_array = scores.scores
    plt.figure(figsize=(6, 6), dpi=150)
    plt.imshow(scores_array["score"], cmap="viridis")
    plt.colorbar(shrink=0.7)
    plt.title("Cosine Greedy spectra similarities" + str(i))
    plt.xlabel("Spectrum #ID")
    plt.ylabel("Spectrum #ID")
    plt.show()
    """

    spectra.clear()
    maxMs2i.clear()
    i= i +1;

triplets=[]

for dupla in duplas:
    triplet = find_triplet(dupla,features_scans,ms2_df)
    triplets_dict = {
        'dupla': dupla,
        'triplet': triplet
    }
    triplets.append(triplets_dict)

file_path = 'triplets.txt'
json_str = json.dumps(triplets, indent=2)

with open(file_path, 'w') as file:
    file.write(json_str)

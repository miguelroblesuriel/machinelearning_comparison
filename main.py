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


def createSpectrum(spectrum_i,spectrum_mz,scan):
    spectrum={
        'i': spectrum_i,
        'mz': spectrum_mz,
        'scan': scan
    }
    return spectrum

input_filename = "049_Blk_Water_NEG.mzMl"
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
    if all_i_norm[i]>=0.2:
        all_i_filtered.append(i)

spectrum_i=[]
spectrum_mz=[]
spectra=[]
actual_scan=all_scan[0]
for i in all_i_filtered:
    if(actual_scan==all_scan[i]):
        spectrum_i.append(all_i_norm[i])
        spectrum_mz.append(all_mz[i])
    else:
        spectra.append(createSpectrum(spectrum_i, spectrum_mz,actual_scan))
        actual_scan=all_scan[i]
        spectrum_i.clear()
        spectrum_mz.clear()
        spectrum_i.append(all_i_norm[i])
        spectrum_mz.append(all_mz[i])

for spectrum in spectra: print(spectrum['scan'])
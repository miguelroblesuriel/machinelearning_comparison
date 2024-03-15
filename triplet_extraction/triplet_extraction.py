import json

import pandas as pd


from pyopenms import *

from massql import msql_fileloading

from comparison.find_triplet import find_triplet

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


def triplet_extraction(input_filename):
    ms1_df, ms2_df = msql_fileloading.load_data(input_filename, cache='feather')
    featurexml_file = "C:/Users/migue/OneDrive/Escritorio/Uni/Beca/machinelearning_comparison/output.featureXML"

    features = FeatureMap()
    FeatureXMLFile().load(featurexml_file, features)

    spectra = []
    i = 0
    duplas = []
    maxMs2i = []
    features_scans = []

    for f in features:
        matched_scans = (ms2_df.loc[(
                                            (f.getMZ() - 0.1 < ms2_df['precmz']) & (ms2_df['precmz'] < f.getMZ() + 0.1)) & (
                                            (f.getRT() / 60 - 0.1 < ms2_df['rt']) & (ms2_df['rt'] < f.getRT() / 60 + 0.1))][
                             'scan'].unique())
        features_scans.append(matched_scans)

        for m in matched_scans:
            maxMs2i.append(ms2_df.loc[ms2_df['scan'] == m]['i'].max())
        Scan_intensity_dict = {
            'scan': matched_scans,
            'maxMs2i': maxMs2i
        }
        Scan_intensity_df = pd.DataFrame(Scan_intensity_dict)
        Scan_intensity_df_sorted = Scan_intensity_df.sort_values(by='maxMs2i')
        scans = Scan_intensity_df_sorted['scan']
        if len(scans) > 1:
            duplas.append(scans[-2:])

        spectra.clear()
        maxMs2i.clear()
        i = i + 1

    triplets = []

    for dupla in duplas:
        print("a")
        triplet = find_triplet(dupla, features_scans, ms2_df)
        triplets_dict = {
            'dupla': dupla,
            'triplet': triplet
        }
        triplets.append(triplets_dict)

    file_path = 'triplets.txt'
    json_str = json.dumps(triplets, indent=2)

    with open(file_path, 'w') as file:
        file.write(json_str)
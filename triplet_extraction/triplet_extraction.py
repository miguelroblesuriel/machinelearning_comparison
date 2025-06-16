import json
import os
import time

import pandas as pd


from pyopenms import *

from massql import msql_fileloading

from comparison.find_triplet import find_triplet

from preprocessing.check_file_quality import check_file_quality
from preprocessing.determine_time_unit import determine_time_unit

from openpyxl import load_workbook

from comparison.compare_two_scans import compare_two_scans

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


def feature_finding(input_filename,input_filepath):
    options = PeakFileOptions()
    options.setMSLevels([1])
    fh = MzMLFile()
    fh.setOptions(options)

    # Load data
    input_map = MSExperiment()
    fh.load(input_filepath, input_map)
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
    output_filename = input_filename.replace(".mzML", "_output.featureXML")
    output_path = os.path.join("C:/Users/usuario/Desktop/Uni/import/machinelearning_comparison/triplet_data/", output_filename)
    fh.store(output_path, features)
    return features


def add_to_excel(excel_file, data_dict, filename):

    if not os.path.exists(excel_file):
        # Create a new Excel file if it doesn't exist
        df = pd.DataFrame(columns=["Metric"])
        df.to_excel(excel_file, index=False, sheet_name="Timing Data")
    # Load the existing workbook
    workbook = load_workbook(excel_file)
    sheet = workbook["Timing Data"]

    # Add the data for each metric
    for metric, value in data_dict.items():
        # Find the row for this metric
        if sheet.cell(row=1, column=1).value == "Metric":
            # Metrics are already written in the first column, so find the row for this metric
            metric_row = None
            for row in range(2, sheet.max_row + 1):
                if sheet.cell(row=row, column=1).value == metric:
                    metric_row = row
                    break
            if not metric_row:
                # If the metric is not found, add it
                metric_row = sheet.max_row + 1
                sheet.cell(row=metric_row, column=1, value=metric)

        # Find the next available column for this file and add the data
        next_col = sheet.max_column + 1
        sheet.cell(row=metric_row, column=next_col, value=value)
    sheet.cell(row=1, column=next_col, value=filename)
    # Save the workbook
    workbook.save(excel_file)


def adjust_time_unit(time_unit,rt):
    if(time_unit=="second"):
        return rt/60
    else:
        return rt

def triplet_extraction(input_filename,input_filepath,threshold,peak_threshold,find_triplet_bool=True,check_quality=True):
    start_time = time.time()
    excel_file = 'C:/Users/usuario/Desktop/Uni/import/machinelearning_comparison/triplet_data/Extra_data.xlsx'
    time_unit=determine_time_unit(input_filepath)
    print(time_unit)
    ms1_df, ms2_df = msql_fileloading.load_data(input_filepath, cache='feather')
    print(ms2_df.columns)
    load_time = time.time()
    featurexml_filename = input_filename.replace(".mzML", "_output.featureXML")

    current_directory = os.getcwd()

    featurexml_file = os.path.join(current_directory, "triplet_data", featurexml_filename)
    print(featurexml_file)
    if os.path.exists(featurexml_file):
        features = FeatureMap()
        FeatureXMLFile().load(featurexml_file, features)
    else:
        features = feature_finding(input_filename, input_filepath)

    find_feature_time = time.time()
    spectra = []
    i = 0
    duplas = []
    maxMs2i = []
    features_scans = []
    features_scans_precmz = []
    print(input_filepath)
    for f in features:
        matched_scans = (ms2_df.loc[(
                                            (f.getMZ() - 0.1 < ms2_df['precmz']) & (ms2_df['precmz'] < f.getMZ() + 0.1)) & (
                                            (f.getRT() / 60 - 0.1 < adjust_time_unit(time_unit,ms2_df['rt'])) & (adjust_time_unit(time_unit,ms2_df['rt']) < f.getRT() / 60 + 0.1))][
                             'scan'].unique())
        if(len(matched_scans)>0):
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
            features_scans_precmz.append(ms2_df[ms2_df['scan'] == scans.iloc[-1]]['precmz'].unique())
        spectra.clear()
        maxMs2i.clear()
        i = i + 1
    find_tuple = time.time()
    if check_quality:
        if len(duplas) > 0:
            scans = []
            for dupla in duplas:
                scans.append(dupla.iloc[0])
            check_file_quality(scans,ms2_df, input_filename)
    check_quality_time = time.time()
    triplets = []
    if find_triplet_bool:
        i = len(duplas)
        for dupla in duplas:
            print(i)
            triplet, comparison_scores = find_triplet(dupla, features_scans, features_scans_precmz, ms2_df,threshold,peak_threshold)
            print(triplet)
            triplets_dict = {
                'dupla': dupla,
                'triplet': triplet,
                'scores' : comparison_scores
            }
            triplets.append(triplets_dict)
            i = i - 1
    find_triplet_time = time.time()
    data_dict = {
        'load_file_time': start_time-load_time,
        'find_feature_time': load_time-find_feature_time,
        'find_tuple_time': find_feature_time-find_tuple,
        'check_quality_time': find_tuple-check_quality_time,
        'find_triplet_time': check_quality_time-find_triplet_time,
        'time_unit': time_unit
    }
    """add_to_excel(excel_file,data_dict,input_filename)"""
    return triplets
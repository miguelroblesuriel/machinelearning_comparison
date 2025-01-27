import numpy as np


from triplet_extraction.triplet_extraction import triplet_extraction
import os

"""input_directorynames = os.listdir("/data/hdd/home/user/LabData/data/GNPS_Library_Provenance/GNPS_Library_Provenance")
for input_directoryname in input_directorynames:
    input_filenames = os.listdir(os.path.join("/data/hdd/home/user/LabData/data/GNPS_Library_Provenance/GNPS_Library_Provenance", input_directoryname))
    for input_filename in input_filenames:
        input_filepath = os.path.join("/data/hdd/home/user/LabData/data/GNPS_Library_Provenance/GNPS_Library_Provenance", input_directoryname,input_filename)
        print(input_filepath)"""
input_filepath = "/home/miguel/Descargas/20240422_BT3_pos_r001.mzML"
input_filename = "20240422_BT3_pos_r001.mzML"
with open('/home/miguel/beca/machinelearning_comparison/triplet_data/read_files.txt', 'r') as file:
    names = file.readlines()
    names = [name.strip() for name in names]
    if input_filename not in names:
        with open('/home/miguel/beca/machinelearning_comparison/triplet_data/read_files.txt','a') as file:
            file.write(input_filename+"\n")
        triplets = triplet_extraction(input_filename,input_filepath, threshold=0.7, peak_threshold=5)
        print("Done triplet extraction")
        output_filename = input_filename.replace(".mzML", "_triplets.npy")
        output_path = os.path.join("/home/miguel/beca/machinelearning_comparison/triplet_data", output_filename)
        np.save(output_path, triplets)


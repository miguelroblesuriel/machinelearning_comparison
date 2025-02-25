import numpy as np
from triplet_extraction.triplet_extraction import triplet_extraction
import os

input_filenames = os.listdir("C:/Users/usuario/Desktop/Uni/import/downloadpublicdata/data/filedownloads/filedownloads_flat_wrong")
for input_filename in input_filenames:
    input_filepath = os.path.join("C:/Users/usuario/Desktop/Uni/import/downloadpublicdata/data/filedownloads/filedownloads_flat_wrong",input_filename)
    with open('C:/Users/usuario/Desktop/Uni/import/machinelearning_comparison/triplet_data/read_files.txt', 'r') as file:
        names = file.readlines()
        names = [name.strip() for name in names]
        if input_filename not in names and input_filename not in [name+"_ms1.msql.feather" for name in names] and input_filename not in [name+"_ms2.msql.feather" for name in names]:
            with open('C:/Users/usuario/Desktop/Uni/import/machinelearning_comparison/triplet_data/read_files.txt', 'a') as file:
                file.write(input_filename + "\n")
            triplets = triplet_extraction(input_filename, input_filepath, threshold=0.7, peak_threshold=5, find_triplet_bool=True,check_quality=True)
            print("Done triplet extraction")
            output_filename = input_filename.replace(".mzML", "_triplets.npy")
            output_path = os.path.join("C:/Users/usuario/Desktop/Uni/import/machinelearning_comparison/triplet_data/", output_filename)
            np.save(output_path, triplets)
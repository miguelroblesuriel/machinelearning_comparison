import numpy as np
import yaml

from triplet_extraction.triplet_extraction import triplet_extraction
import os


with open('Config.yaml', 'r') as file:
    yaml_data = yaml.safe_load(file)
input_filenames = os.listdir(yaml_data['input_dir_route'])
for input_filename in input_filenames:
    try:
        input_filepath = os.path.join(yaml_data['input_dir_route'],input_filename)
        with open(os.path.join(yaml_data['output_dir_route'],'read_files.txt'), 'r') as file:
            names = file.readlines()
            names = [name.strip() for name in names]
            if input_filename not in names and input_filename not in [name+"_ms1.msql.feather" for name in names] and input_filename not in [name+"_ms2.msql.feather" for name in names]:
                with open(os.path.join(yaml_data['output_dir_route'],'read_files.txt'), 'a') as file:
                    file.write(input_filename + "\n")
                triplets = triplet_extraction(input_filename, input_filepath, threshold=yaml_data['comparison_threshold'], peak_threshold=yaml_data['peak_threshold'], find_triplet_bool=yaml_data['find_triplet'],check_quality=yaml_data['check_quality'])
                print("Done triplet extraction")
                output_filename = input_filename.replace(".mzML", "_triplets.npy")
                output_path = os.path.join(yaml_data['output_dir_route'], output_filename)
                np.save(output_path, triplets)
    except  MemoryError:
        print("Error de memoria")


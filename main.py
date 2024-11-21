import numpy as np
from triplet_extraction.triplet_extraction import triplet_extraction
import os


input_filenames = os.listdir(os.path.join(os.getcwd(), "/mzmls"))
for input_filename in input_filenames:
    triplets = triplet_extraction(input_filename, threshold=0.75, peak_threshold=10)
    print("Done triplet extraction")
    output_filename = input_filename.replace(".mzML", "_triplets.npy")
    np.save(output_filename, triplets)


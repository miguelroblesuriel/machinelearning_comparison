import numpy as np
from triplet_extraction.triplet_extraction import triplet_extraction


input_filename = "G75229_1x_10ul_RA7_01_19775.mzMl"
triplets = triplet_extraction(input_filename)
output_filename = input_filename.replace(".mzMl", "_triplets.npy")
np.save(output_filename, triplets)
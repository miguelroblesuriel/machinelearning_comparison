import numpy as np
from triplet_extraction.triplet_extraction import triplet_extraction


input_filename = "230421-Exp-PAN-1-110-SMGA-MSMS1.mzMl"
triplets = triplet_extraction(input_filename)
output_filename = input_filename.replace(".mzMl", "_triplets.npy")
np.save(output_filename, triplets)
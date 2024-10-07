from matchms import Spectrum
import numpy as np

def to_float(array):
    array_float = []
    for element in array:
        array_float.append(float(element))
    return np.array(array_float)

def createSpectrum(spectrum_i, spectrum_mz, precursor_mz=0):
    '''

    :param spectrum_i:
    :param spectrum_mz:
    :return:
    '''

    spectrum = Spectrum(mz=to_float(spectrum_mz),
                        intensities=to_float(spectrum_i),
                        metadata={'precursor_mz': float(precursor_mz)},
                        metadata_harmonization=None)
    return spectrum
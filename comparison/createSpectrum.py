from matchms import Spectrum

def createSpectrum(spectrum_i, spectrum_mz, precursor_mz):
    '''

    :param spectrum_i:
    :param spectrum_mz:
    :return:
    '''
    spectrum = Spectrum(mz=spectrum_mz,
                        intensities=spectrum_i,
                        metadata={'precursor_mz': float(precursor_mz)},
                        metadata_harmonization=None)
    return spectrum
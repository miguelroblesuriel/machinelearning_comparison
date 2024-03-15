from matchms import Spectrum

def createSpectrum(spectrum_i, spectrum_mz):
    '''

    :param spectrum_i:
    :param spectrum_mz:
    :return:
    '''
    spectrum = Spectrum(mz=spectrum_mz,
                        intensities=spectrum_i,
                        metadata=None,
                        metadata_harmonization=None)
    return spectrum
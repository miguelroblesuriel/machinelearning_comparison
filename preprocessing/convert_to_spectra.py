from comparison.createSpectrum import createSpectrum


def convert_to_spectra(tensors,tensor_padding_masks):
    spectra = []
    for tensor,mask in zip(tensors,tensor_padding_masks.tolist()):
        padded_spectrum_mz = [tuple[0].item() for tuple in tensor]
        padded_spectrum_i = [tuple[1].item() for tuple in tensor]
        i = 0
        depadded_spectrum_mz = []
        depadded_spectrum_i = []
        for condition in mask:
            if not condition:
                depadded_spectrum_mz.append(padded_spectrum_mz[i])
                depadded_spectrum_i.append(padded_spectrum_i[i])
            i = i + 1

        depadded_spectrum_i = depadded_spectrum_i[1:]
        depadded_spectrum_mz = depadded_spectrum_mz[1:]
        spectrum = createSpectrum(depadded_spectrum_i, depadded_spectrum_mz)
        spectra.append(spectrum)
    return spectra
from comparison.cosine_greedy import cosine_greedy
from comparison.createSpectrum import createSpectrum
from preprocessing.peak_processing import peak_processing
import numpy
import matplotlib.pyplot as plt
def check_file_quality(scans,ms2_df,input_filename):
    spectra = []
    filtered_spectra = []
    valid_spectra= []
    oneD_scores = []
    for scan in scans:
        spectra.append(createSpectrum(ms2_df[ms2_df['scan'] == scan]['i_norm'].to_numpy(),
                                      numpy.sort(ms2_df[ms2_df['scan'] == scan]['mz'].to_numpy()),
                                      ms2_df[ms2_df['scan'] == scan]['precmz'].unique(), scan))
    for s in spectra:
        filtered_spectra.append(peak_processing(s))

    for s1 in filtered_spectra:
        for s2 in filtered_spectra:
            if s1 != s2 and s1.get("precursor_mz") != s2.get("precursor_mz"):
                valid_spectra.append(s2)
        if valid_spectra:
            print(s1)
            print(valid_spectra)
            spectrum_dupla = []
            spectrum_dupla.append(s1)
            spectrum_dupla.append(s1)
            scores = cosine_greedy(0.005,spectrum_dupla,valid_spectra)
            scores_array = scores.scores.to_array()
            oneD_scores.extend(scores_array["CosineGreedy_score"].flatten())
            valid_spectra = []
    counter = 0
    for score in oneD_scores:
        print(score)
        if score > 0.7 and score != 1:
            counter = counter + 1

    if len(oneD_scores)>0:
        if counter/2 > 40:
            with open("C:/Users/usuario/Desktop/Uni/machinelearning_comparison/Mzml_utiles_cantidad.txt", "a") as f:
                f.write(input_filename)
        if (counter/2)/len(oneD_scores) > 0.1:
            with open("C:/Users/usuario/Desktop/Uni/machinelearning_comparison/Mzml_utiles_calidad.txt", "a") as f:
                f.write(input_filename)
        if counter/2 > 40 and (counter/2)/len(oneD_scores) > 0.1:
            with open("C:/Users/usuario/Desktop/Uni/machinelearning_comparison/Mzml_utiles_calidad_y_cantidad.txt", "a") as f:
                f.write(input_filename)

    #scores = cosine_greedy(0.005, spectra, spectra)
    #scores_array = scores.scores.to_array()
    #oneD_scores = scores_array["CosineGreedy_score"].flatten()
    print(oneD_scores)
    plt.hist(oneD_scores, bins=50, range=(0, 1), edgecolor='black', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(input_filename)
    plt.show()
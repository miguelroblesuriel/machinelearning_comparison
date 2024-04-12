import numpy
import os
import gensim
from massql import msql_fileloading
from comparison.createSpectrum import createSpectrum
from comparison.cosine_greedy import cosine_greedy
from comparison.modified_cosine import modified_cosine
from comparison.spec2vec import spec2vec
from comparison.ms2deepscore import ms2deepscore
from visualization.mirror_plot import mirror_plot
from visualization.plot_scores import plot_scores
from ms2deepscore.models import load_model
from spec2vec import SpectrumDocument
from spec2vec.model_building import train_new_word2vec_model

input_filename = "049_Blk_Water_NEG.mzMl"
ms1_df, ms2_df = msql_fileloading.load_data(input_filename, cache='feather') #importar el archivo con la información del experimento
file_path = 'triplets.npy'
loaded_data = numpy.load(file_path, allow_pickle=True)#importar los sets de dupla/tripletas
modelo_MS2DeepScore = load_model("MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5")



model_file = 'modelo.model'




dupla = loaded_data[0]["dupla"]
tripletas = loaded_data[0]["triplet"]

spectra_dupla = [createSpectrum(ms2_df[ms2_df['scan'] == scan_dupla]['i_norm'].to_numpy(),
                numpy.sort(ms2_df[ms2_df['scan'] == scan_dupla]['mz'].to_numpy()), ms2_df[ms2_df['scan'] == scan_dupla]['precmz'].unique()) for scan_dupla in dupla]#creación de los espectros de las duplas
spectra_tripletas = [createSpectrum(ms2_df[ms2_df['scan'] == scan_tripleta]['i_norm'].to_numpy(),
                numpy.sort(ms2_df[ms2_df['scan'] == scan_tripleta]['mz'].to_numpy()),ms2_df[ms2_df['scan'] == scan_tripleta]['precmz'].unique()) for scan_tripleta in tripletas]#creación de los espectros de las tripletas


spectrum_documents = [SpectrumDocument(s, n_decimals=2) for s in spectra_tripletas]
modelo_Spec2Vec = train_new_word2vec_model(spectrum_documents, iterations=[25], filename=model_file, workers=2,
                                     progress_logger=True, vector_size=300)

mirror_plot(spectra_dupla[0],spectra_dupla[1]) #Representa los dos espectros enfrentados
scores = cosine_greedy(0.005, spectra_dupla, spectra_tripletas)#compara los espectros utilizando el cosineGreedy
scores_array = scores.scores
print(scores.scores["matches"])
plot_scores((scores.scores)["score"]) #Representa los valores de la comparación
scores = modified_cosine(0.005, spectra_dupla, spectra_tripletas) #compara los espectros utilizando el modifiedCosine
plot_scores(scores.scores["score"])
scores = ms2deepscore(modelo_MS2DeepScore, spectra_dupla, spectra_tripletas)#compara los espectros utilizando MS2DeepScore
plot_scores(scores.scores)
scores = spec2vec(modelo_Spec2Vec, spectra_dupla, spectra_tripletas)#compara los espectros utilizando Spec2Vec
plot_scores(scores.scores)
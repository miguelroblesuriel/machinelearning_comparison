import numpy
from machinelearning_model.CustomSpectraDataset import CustomSpectraDataset

file_path = '049_Blk_Water_NEG_triplets.npy'
loaded_data = numpy.load(file_path, allow_pickle=True)#importar los sets de dupla/tripletas
duplas = []
triplets = []
scores = []
for item in loaded_data:
    if item["triplet"] != []:
        duplas.append(item["dupla"].tolist())
        triplets.append(item["triplet"])
        scores.append(item["scores"])

dataset = CustomSpectraDataset(duplas, triplets, scores)

anchor, dupla, triplet = dataset[300]
print(anchor)
print(dupla)
print(triplet)

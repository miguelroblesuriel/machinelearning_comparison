from massql import msql_fileloading
import pandas as pd
input_filepath = "C:/Users/usuario/Desktop/Uni/import/downloadpublicdata/data/filedownloads/filedownloads_flat_wrong/160528_FHS_Eicosanoid_Plate_13_39.mzML"
ms1_df, ms2_df = msql_fileloading.load_data(input_filepath, cache='feather')
"""print(ms2_df.drop_duplicates(subset=['scan'])['rt'])"""
print(ms2_df['precmz'].unique())

"""371.6582281113849 597.3050166950882
   352.13279695492497 568.3624668067964
143.49447728595842 380.248921201859"""
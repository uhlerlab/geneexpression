from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.decomposition import IncrementalPCA
import pandas as pd

cloneAnnotation = np.load('clone_annotation_in_vitro.npz')
cellMetadataInVitroDay = np.loadtxt(
    'cell_metadata_in_vitro.txt', skiprows=1, usecols=(0,))
metadata = pd.read_csv('cell_metadata_in_vitro.txt', sep='\\t', header=0)
cellMetadataInVitroType = np.genfromtxt(
    'cell_metadata_in_vitro.txt', dtype='str',  skip_header=1, usecols=(2,))
day4_6 = metadata.loc[metadata['Time point'] > 3]
day4_6_neutrophil = day4_6.loc[(day4_6['Annotation'] == 'Neutrophil')]
day4_6_monocyte = day4_6.loc[(day4_6['Annotation'] == 'Monocyte')]
clone_data = csc_matrix(
    (cloneAnnotation['data'], cloneAnnotation['indices'], cloneAnnotation['indptr']), shape=(130887, 5864)).toarray()
row_indices = []
f = open("tmaps.txt")
content = f.readlines()
col_indices = content[0].split("\t")
col_indices = col_indices[1:]
for i in range(len(col_indices)):
    col_indices[i] = int(col_indices[i][5:])
content = content[1:]
for i in range(len(content)):
    x = content[i].split("\t")
    row_indices.append(int(x[0][5:]))
    content[i] = x[1:]
tmaps = np.array(content)
y = []
# print(cellMetaDataInVitroType)
for i in range(len(col_indices)):
    if cellMetadataInVitroType[col_indices[i]] == 'Monocyte':
        y.append(0)

    if cellMetadataInVitroType[col_indices[i]] == 'Neutrophil':
        y.append(1)

totalCorrect = 0

for r in range(len(row_indices)):
    monocyte_total = 0
    neutrophil_total = 0
    for c in range(len(col_indices)):
        if y[c] == 0:
            monocyte_total += float(tmaps[r][c])
        else:
            neutrophil_total += float(tmaps[r][c])
    wot_calc = "Neutrophil" if neutrophil_total > monocyte_total else "Monocyte"
    clone_index = np.where(clone_data[row_indices[r]] == 1)[0]
    cell_index = metadata.loc[np.where(
        clone_data[:, clone_index] == 1)[0]]
    m_count= 0
    n_count= 0
    for row in cell_index.iterrows():
        if row[1]['Time point'] > 3 and row[1]['Annotation'] == 'Neutrophil':
            n_count += 1
        if row[1]['Time point'] > 3 and row[1]['Annotation'] == 'Monocyte':
            m_count += 1
    actual_calc= "Neutrophil" if n_count > m_count else "Monocyte"
    if actual_calc == wot_calc:
        totalCorrect += 1

performance= totalCorrect/len(row_indices)
print(performance)

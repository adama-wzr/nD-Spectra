import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import umap
import umap.plot
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_avg_Raman(nClusters, kMeansLabels, normIntensity):
    # get data size and shape
    nWavelength, nData = normIntensity.shape
    # Array for cluster size
    clusterSize = np.zeros(nClusters)
    clusterAvgRaman = np.zeros((nWavelength, nClusters))

    # add all raman to their respective clusters

    for n in range(nData):
        index = kMeansLabels[n]
        clusterSize[n]+=1
        clusterAvgRaman[:,index] = np.add(clusterAvgRaman[:,index], normIntensity[:, n])
    
    # normalize clusters for average
    



Na_EO_1_4_path = f'/home/guang/Documents/PEO-TFSI/Andre_Results/Na_EO_1over12/'
filename = f'20171010-natfs-peo-1over12-lev26-.txt'

full_path = os.path.join(Na_EO_1_4_path, filename)

df1 = pd.read_csv(full_path, sep='\t', low_memory=False)

raw_data = df1.values

raw_values = raw_data[1::,1::].astype(float)
RamanShift = raw_data[1::, 0]

# clean the memory

del(df1)
del(raw_data)

# Normalizing 

nInt, nRaman = raw_values.shape
# print(raw_values.shape)

norm_Intensity = np.zeros_like(raw_values)

for i in range(nRaman):
    norm_Intensity[:,i] = raw_values[:,i]/np.sum(raw_values[:,i])
    # print(np.sum(norm_Intensity[:,i]))

# print(norm_Intensity)

# Transpose matrix

data_to_fit = norm_Intensity.T

# Import data into UMAP

my_map = umap.UMAP(
    min_dist=0.001,
    n_neighbors=15,
    metric='correlation',
    random_state=42
).fit(data_to_fit)

# plot
# umap.plot.points(my_map)
# plt.show()

coords = my_map.embedding_

# # kMeans Clustering on UMAP

# K = range(2, 20)
# # fits = []
# score = []


# for k in K:
#     localScore = 0
#     for r in range(10):
#         # train the model for current value of k on training data
#         model = KMeans(n_clusters = k, random_state = r, n_init='auto').fit(coords)
        
#         # append the model to fits
#         # fits.append(model)
        
#         # Append the silhouette score to scores
#         localScore += silhouette_score(coords, model.labels_, metric='euclidean')
#         # score.append(silhouette_score(coords, model.labels_, metric='euclidean'))
#         # if r % 10 == 0:
#         #     print(localScore)
#     score.append(localScore)
#     print(k)
#     print(localScore)


# sns.lineplot(x = K, y = score)
# np.savetxt('test.csv', coords, delimiter=',')

# actual kMeans

k = 7
kmeans = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(coords)
np.savetxt('k7_labels.csv', kmeans.labels_, delimiter=',')

sns.scatterplot(x=coords[:,0], y=coords[:,1], hue=kmeans.labels_)

plt.show()
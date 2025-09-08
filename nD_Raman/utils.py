'''
Utils

Authors:
    - Andre Adam
    - 
List of Major Updates (date):

    - Sep 4th, 2025 --> project created
'''
import pandas as pd
import numpy as np
import os
import umap
import umap.plot
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


'''

Block for saving outputs

'''

def saveUMAP_coords(my_coords, filename):
    '''
    Function saveUMAP_coords:
    Inputs:
        - (ndarray) saveUMAP_coords
    Outputs:
        - none
    
    Function will create the .csv 'filename'
    and save the umap coords to it.
    '''

    np.savetxt(filename, my_coords, delimiter=',', header='x,y')

    return

'''

Block for normalizations and other data
manipulation pre-processing steps

'''

def get_avg_Raman(nClusters, kMeansLabels, normIntensity):
    '''
    Function get_avg_Raman:
    Inputs:
        - (int) number of clusters
        - (ndarray) kMeans Labels
        - (ndarray) normalized Intensity  
    Outputs:
        - (ndarray) average raman for each cluster.

    '''
    # get data size and shape
    nWavelength, nData = normIntensity.shape
    # Array for cluster size
    clusterSize = np.zeros(nClusters)
    clusterAvgRaman = np.zeros((nWavelength, nClusters))

    # add all raman to their respective clusters

    for n in range(nData):
        index = kMeansLabels[n]
        clusterSize[index]+=1
        clusterAvgRaman[:,index] = np.add(clusterAvgRaman[:,index], normIntensity[:, n])
    
    # normalize clusters for average
    for n in range(nClusters):
        clusterAvgRaman[:, n] = np.divide(clusterAvgRaman[:,n], clusterSize[n])
    
    # return normalize cluster labels
    return clusterAvgRaman

'''

Block for reading input data + Mapping

'''

def readInput(readPath):
    df1 = pd.read_csv(readPath, sep='\t', low_memory=False)
    raw_data = df1.values

    raw_values = raw_data[1::,1::].astype(float)
    Wavelength = raw_data[1::, 0].astype(float)

    # clean the memory

    del(df1)
    del(raw_data)

    # Normalizing 

    nInt, nRaman = raw_values.shape
    # print(raw_values.shape)

    norm_Intensity = np.zeros_like(raw_values)

    for i in range(nRaman):
        norm_Intensity[:,i] = raw_values[:,i]/np.sum(raw_values[:,i])
    
    # Transpose matrix

    data_to_fit = norm_Intensity.T

    return Wavelength, data_to_fit
    



def SpatialUMAP(Wavelength, data_to_fit, saveFlag, savePath, UMAP_nn, UMAP_minDist, UMAP_metric, randStat):
    '''
    Function SpatialUMAP:
    Inputs:
        - (ndarray) Wavelength data
        - (ndarray) data_to_fit
        - (bool) flag to save coordinates from UMAP
        - full save path
        - (int) number of neighbors UMAP
        - (float) min_dist UMAP
        - (string) metric for UMAP
        - random state
    Outputs:
        - UMAP embedding
        - Wavelength (array of Raman Shift)
        - Normalized Intensities
    '''

    # Import data into UMAP

    my_map = umap.UMAP(
        min_dist=UMAP_minDist,
        n_neighbors=UMAP_nn,
        metric=UMAP_metric,
        random_state=randStat
    ).fit(data_to_fit)

    # if save coords, save coords

    if saveFlag:
        coords = my_map.embedding_
        header = 'x,y'
        np.savetxt(savePath, coords, delimiter=',', header='x,y')

    # return UMAP
    return my_map
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
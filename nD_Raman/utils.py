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

import plot


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

def get_avg_Raman(nClusters, kMeansLabels, normIntensity, saveFlag, saveName, Shift):
    '''
    Function get_avg_Raman:
    Inputs:
        - (int) number of clusters
        - (ndarray) kMeans Labels
        - (ndarray) normalized Intensity 
        - (bool) save data
        - (string) save file name
        - (ndarray) Raman shift
    Outputs:
        - (ndarray) average raman for each cluster.

    '''
    # get data size and shape
    nWavelength, nData = normIntensity.shape
    # Array for cluster size
    clusterSize = np.zeros(nClusters)
    clusterAvgRaman = np.zeros((nWavelength, nClusters ))

    # add all raman to their respective clusters

    for n in range(nData):
        index = kMeansLabels[n]
        clusterSize[index]+=1
        clusterAvgRaman[:,(index )] = np.add(clusterAvgRaman[:,(index)], normIntensity[:, n])
    
    # normalize clusters for average
    for n in range(nClusters):
        clusterAvgRaman[:, n] = np.divide(clusterAvgRaman[:,n], clusterSize[n])

    # save array

    if saveFlag:
        array = np.zeros((nWavelength, nClusters+1))
        array[:,0] = Shift
        array[:, 1::] = clusterAvgRaman
        header = 'RamanShift,'
        for i in range(nClusters):
            header += str(i)
            if i != (nClusters-1):
                header += ','
        np.savetxt(saveName, array, delimiter=',', header=header)
    
    # return normalize cluster labels
    return clusterAvgRaman

'''

Block for reading input data + Mapping

'''

def readInput(readPath):
    df1 = pd.read_csv(readPath, sep='\t', low_memory=False)
    raw_data = df1.values

    raw_values = raw_data[1::,1::].astype(float)
    RamanShift = raw_data[1::, 0].astype(float)

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

    return RamanShift, data_to_fit
    



def SpatialUMAP(RamanShift, data_to_fit, saveFlag, savePath, UMAP_nn, UMAP_minDist, UMAP_metric, randStat):
    '''
    Function SpatialUMAP:
    Inputs:
        - (ndarray) RamanShift data
        - (ndarray) data_to_fit
        - (bool) flag to save coordinates from UMAP
        - full save path
        - (int) number of neighbors UMAP
        - (float) min_dist UMAP
        - (string) metric for UMAP
        - random state
    Outputs:
        - UMAP embedding
        - RamanShift (array of Raman Shift)
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


'''

Clustering Related Material

'''

def kMeansOpt(my_coords, K, rNum, filename, savepath, verbose):
    '''
    Function kMeans Opt:
    Inputs:
        - (ndarray) my_coords: UMAP coordinates
        - (range) K number of clusters
        - (int) number of random states to check 
        - (string) save file name
        - (string) save path
        - (bool) verbose
    Outputs:
        - none.
    
    Function will perform number of cluster optimization
    for K-Means and save data accordingly.
    '''

    score = []

    for k in K:
        localScore = 0
        for r in range(rNum):
            # train model for current value of k
            model = KMeans(n_clusters=k, random_state=r, n_init='auto').fit(my_coords)

            # Append Silhouette scores to score
            localScore += silhouette_score(my_coords, model.labels_, metric='euclidean')
        
        score.append(localScore)
        if verbose:
            print(k)
            print(localScore)

    # plot k-Opt
    plot.plotKMeansOpt(K, score, filename)

    return


def singleKMeans(myCoords, k, random_seed):
    '''
    singleKMeans Function:
    Inputs:
        - (ndarray) myCoords, UMAP coordinates
        - (int) number of clusters k
        - random seed for the random state
    Outputs:
        - output cluster labels from sklearn KMeans
    '''
    
    kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init='auto').fit(myCoords)
    
    return kmeans

def AvgSpatialRaman(RamanShift, RamanSpectra, nSamples, filename):
    '''
    AvgSpatialRaman:
    Inputs:
        - (ndarray) Raman shift
        - (ndarray) Spatial Raman spectra
        - (int) number of samples
        - (str) filename for saving
    Outputs:
        - none.

    Function will save the array to a csv file
    '''

    nShift = len(RamanShift)

    avgSpectra = np.zeros((nShift, nSamples+1))
    avgSpectra[:,0] = RamanShift

    NR, NC = RamanSpectra.shape

    n_data = int(NR/nSamples)

    header = "RamanShift"

    for i in range(nSamples):
        header += ','
        avgSpectra[:,i+1] = np.average(RamanSpectra[n_data*i:n_data*(i+1),:], axis=0)
        header += str(i)


    np.savetxt(filename, avgSpectra, delimiter=',', header=header)


    return
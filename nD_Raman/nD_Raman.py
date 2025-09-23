'''
Main Project File.

I write most of my code in C/C++, so
please forgive my stylistic choices.

09/08/2025 - I just found out you don't
need 'return' at the end of functions. But
I will do it anyways, it makes me feel
safer.

Authors:
    - Andre Adam
    - 
List of Major Updates (date):

    - Sep 4th, 2025 --> project created
    - Sep 22nd, 2025 --> multi input

'''

import os
import numpy as np
import pandas
import matplotlib.pyplot as plt

import plot
import utils

def multiInput_func():
    '''
        This function will handle mapping multiple images as
        input all at once.
    '''

    verbose = True

    # Paths

    dataPath = f'/home/guang/Documents/PEO-TFSI/Andre_Results/Batch_Test'
    
    nInputs = 4

    inputName = []

    inputName.append(f'purePEO_Raman.txt')
    inputName.append(f'PEO1over4_Raman.txt')
    inputName.append(f'PEO1over12_Raman.txt')
    inputName.append(f'PEO1over23_Raman.txt')
    

    savePath = f'/home/guang/Documents/PEO-TFSI/Andre_Results/Batch_Test/Test_Out'

    # random state seed
    random_seed = 42 # the answer to everything
    # Show Plots
    showPlots = True

    # UMAP related stuff

    runUMAP = True
    plotUMAP = True
    saveUMAP = True
    saveCoords = True
    saveUMAP_K = True

    UMAP_nn = 5
    UMAP_minDist = 0.001
    UMAP_metric = 'correlation'

    umapCoordsName = str()
    umapSaveName = str()
    umapSaveKName = str()

    if saveUMAP:
        # has to be png or jpg
        umapSaveName = f'TestUMAP.jpg'
    
    if saveCoords:
        # str has to conform to np.savetxt
        umapCoordsName = f'TestUMAP.csv'

    if saveUMAP_K:
        #str has to be png or jpg
        umapSaveKName = f'TestUMAP_Kmeans.jpg'

    # Clustering Related Variables

    kMeansOpt = False
    kOptBounds = range(2, 20)
    nSeeds = 25

    kMeansOptName = f'kMeansOpt_multi4'
    
    k = 8 # provisory k if Opt == False

    # read data

    RamanShift = []
    dataToFit = []
    norm_Intensity= []

    for n in range(nInputs):
        temp1, temp2 = utils.readInput(os.path.join(dataPath, inputName[n]))
        # append
        RamanShift.append(temp1)
        dataToFit.append(temp2)
        norm_Intensity.append(dataToFit[n].T)

    # concatenate all the data
    nData, DataLength = dataToFit[0].shape

    concatData = np.zeros((nData*nInputs, DataLength))

    for n in range(nInputs):
        concatData[nData*n:nData*(n+1),:] = dataToFit[n]

    # call UMAP
    umap_results = None
    coords = None

    if runUMAP:
        umap_results = utils.SpatialUMAP(
            RamanShift,
            concatData,
            saveCoords,
            os.path.join(savePath, umapCoordsName),
            UMAP_nn,
            UMAP_minDist,
            UMAP_metric,
            random_seed
            )
        coords = umap_results.embedding_

    # check

    if type(coords) == 'NoneType':
        print("UMAP has failed. Please try again...")
        return

    # save umap related stuff

    if saveCoords:
        utils.saveUMAP_coords(coords, umapCoordsName)

    # plot umap

    if plotUMAP:
        plot.plotUMAP(umap_results, saveUMAP, umapSaveName)

    # Plotting/Output Related Info

    plotMeanRaman = True
    meanRamanName = "MeanRaman_multi.jpg"
    plotMeanRamanOffset = True
    offsetNumber = 0.01
    offsetRamanName = "OffsetRaman_multi.jpg"

    saveClusterRaman = True
    ClusterRamanName = "ClusterRaman.csv"

    plotRamanSub = True
    RamanSubName = "SubRaman_multi.jpg"

    plotRecRaman = True
    RamanRecName = "RamanRec_multi.jpg"

    # k-Means
    kMeans = utils.singleKMeans(coords, k, random_seed)

    # average cluster Raman

    clusterAvgRaman = utils.get_avg_Raman(k, kMeans.labels_, concatData.T, saveClusterRaman, ClusterRamanName, RamanShift[0])

    # plotting

    if plotRamanSub:
        plot.plotMeanRaman_sub(k, clusterAvgRaman, RamanShift[0], RamanSubName)
    
    if plotMeanRaman:
        plot.plotMeanRaman(k, clusterAvgRaman, RamanShift[0], meanRamanName)
    
    if plotMeanRamanOffset:
        plot.plotMeanRaman_stack(k, clusterAvgRaman, RamanShift[0], offsetRamanName)
    
    if plotRecRaman:
        for n in range(nInputs):
            RamanRecName = f'RamanRec_multi{n}.jpg'
            plot.reconstructLabels(kMeans.labels_[n*nData:(n+1)*nData], inputName[n], RamanRecName)

    if saveUMAP_K:
        plot.plotUMAP_Clusters(coords, kMeans.labels_, umapSaveKName)

    # finally, if show-plot = true
    if showPlots:
        plt.show()

    return

def main():
    '''
    Main function simply reads user input and does whatever
    is asked to do.
    '''
    multiInput = True

    if multiInput:
        multiInput_func()
        return

    verbose = True

    # Paths

    dataPath = f'/home/guang/Documents/PEO-TFSI/Andre_Results/Na_EO_1over23/'
    inputName = f'NaTFS-PEO-1OVER23-100by100um-lev26-_008_Spec.Data 1_F (B+R) (Sub BG) (SG).txt'

    savePath = f'/home/guang/Documents/PEO-TFSI/Test_Out/'

    # Expected Outputs (booleans and file names)
    '''
    This setup code will be delegated to a function in
    the near future.

    I need to put most of this stuff on a data
    structure or a class, but for now this is the way.

    I will also come up with a GUI for this, and then 
    auto-populate the class. Open to suggestions
    '''
    # random state seed
    random_seed = 42 # the answer to everything
    # Show Plots
    showPlots = True

    # UMAP related stuff

    runUMAP = True
    plotUMAP = True
    saveUMAP = True
    saveCoords = True

    UMAP_nn = 5
    UMAP_minDist = 0.001
    UMAP_metric = 'correlation'

    umapCoordsName = str()
    umapSaveName = str()

    if saveUMAP:
        # has to be png or jpg
        umapSaveName = "TestUMAP_1o23"

    # plotting

    if plotRamanSub:
        plot.plotMeanRaman_sub(k, clusterAvgRaman, RamanShift, RamanSubName)
    
    if plotMeanRaman:
        plot.plotMeanRaman(k, clusterAvgRaman, RamanShift, meanRamanName)
    
    if plotMeanRamanOffset:
        plot.plotMeanRaman_stack(k, clusterAvgRaman, RamanShift, offsetRamanName)
    
    if plotRecRaman:
        plot.reconstructLabels(kMeans.labels_, RamanRecName)
    
    if saveCoords:
        # str has to conform to np.savetxt
        umapCoordsName = "TestUMAPCoords_1o23.csv"

    # Clustering Related Variables

    kMeansOpt = False
    kOptBounds = range(2, 12)
    nSeeds = 50

    kMeansOptName = f'kMeansOpt_1o23'
    
    k = 4 # provisory k if Opt == False

    # Plotting/Output Related Info

    plotMeanRaman = True
    meanRamanName = "MeanRaman_1o23.jpg"
    plotMeanRamanOffset = True
    offsetNumber = 0.01
    offsetRamanName = "OffsetRaman_1o23.jpg"

    saveClusterRaman = True
    ClusterRamanName = "ClusterRaman.csv"

    plotRamanSub = True
    RamanSubName = "SubRaman_1o23.jpg"

    plotRecRaman = True
    RamanRecName = "RamanRec_1o23.jpg"
    
    '''
    
    END OF INPUT BLOCK

    '''

    # read data

    RamanShift, dataToFit = utils.readInput(os.path.join(dataPath, inputName))

    norm_Intensity = dataToFit.T

    # call UMAP
    umap_results = None
    coords = None

    if runUMAP:
        umap_results = utils.SpatialUMAP(
            RamanShift,
            dataToFit,
            saveCoords,
            os.path.join(savePath, umapCoordsName),
            UMAP_nn,
            UMAP_minDist,
            UMAP_metric,
            random_seed
            )
        coords = umap_results.embedding_

    # check

    if type(coords) == 'NoneType':
        print("UMAP has failed. Please try again...")
        return

    # save umap related stuff

    if saveCoords:
        utils.saveUMAP_coords(coords, umapCoordsName)

    # plot umap

    if plotUMAP:
        plot.plotUMAP(umap_results, saveUMAP, umapSaveName)

    # kMeans -> find optimal k
    if kMeansOpt:
        utils.kMeansOpt(coords, kOptBounds, nSeeds, kMeansOptName, savePath, verbose)
    
    # k-Means
    kMeans = utils.singleKMeans(coords, k, random_seed)

    # average cluster Raman

    clusterAvgRaman = utils.get_avg_Raman(k, kMeans.labels_, norm_Intensity, saveClusterRaman, ClusterRamanName, RamanShift[0])

    # plotting

    if plotRamanSub:
        plot.plotMeanRaman_sub(k, clusterAvgRaman, RamanShift, RamanSubName)
    
    if plotMeanRaman:
        plot.plotMeanRaman(k, clusterAvgRaman, RamanShift, meanRamanName)
    
    if plotMeanRamanOffset:
        plot.plotMeanRaman_stack(k, clusterAvgRaman, RamanShift, offsetRamanName)
    
    if plotRecRaman:
        plot.reconstructLabels(kMeans.labels_, inputName, RamanRecName)


    # finally, if show-plot = true
    if showPlots:
        plt.show()

    return



if __name__ == "__main__":
    main()

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

'''

import os
import numpy as np
import pandas
import matplotlib.pyplot as plt

import plot
import utils

def main():
    '''
    Main function simply reads user input and does whatever
    is asked to do.
    '''
    verbose = True
    # Paths

    dataPath = f'/home/guang/Documents/PEO-TFSI/Bootstrap_PEO_TFSI_GY_2018/Pure PEO/'
    inputName = f'180802-PEO-NO-SALT-LEV-26_035_TABLE.txt'

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
        umapSaveName = "TestUMAP_PurePEO.jpg"
    
    if saveCoords:
        # str has to conform to np.savetxt
        umapCoordsName = "TestUMAPCoords_PurePEO.csv"

    # Clustering Related Variables

    kMeansOpt = True
    kOptBounds = range(2, 12)
    nSeeds = 50

    kMeansOptName = f'kMeansOpt'
    
    k = 4 # provisory k if Opt == False

    # Plotting/Output Related Info

    plotMeanRaman = True
    meanRamanName = "MeanRaman_PurePEO.jpg"
    plotMeanRamanOffset = True
    offsetNumber = 0.01
    offsetRamanName = "OffsetRaman_PurePEO.jpg"

    plotRamanSub = True
    RamanSubName = "SubRaman_PurePEO.jpg"

    plotRecRaman = True
    RamanRecName = "RamanRec_PurePEO.jpg"
    
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

    clusterAvgRaman = utils.get_avg_Raman(k, kMeans.labels_, norm_Intensity)

    # plotting

    if plotRamanSub:
        plot.plotMeanRaman_sub(k, clusterAvgRaman, RamanShift, RamanSubName)
    
    if plotMeanRaman:
        plot.plotMeanRaman(k, clusterAvgRaman, RamanShift, meanRamanName)
    
    if plotMeanRamanOffset:
        plot.plotMeanRaman_stack(k, clusterAvgRaman, RamanShift, offsetRamanName)
    
    if plotRecRaman:
        plot.reconstructLabels(kMeans.labels_, RamanRecName)


    # finally, if show-plot = true
    if showPlots:
        plt.show()

    return



if __name__ == "__main__":
    main()
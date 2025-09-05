'''
Main Project File.

Authors:
    - Andre Adam
    - 
List of Major Updates (date):

    - Sep 4th, 2025 --> project created

'''

import os
import numpy as np
import pandas
import matplotlib as plt

import plot
import utils

def main():
    '''
    Main function simply reads user input and does whatever
    is asked to do.
    '''
    # Paths

    dataPath = f'/home/guang/Documents/PEO-TFSI/Andre_Results/Na_EO_1over12/'
    inputName = f'20171010-natfs-peo-1over12-lev26-.txt'

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
        umapSaveName = "TestUMAP.jpg"
    
    if saveCoords:
        # str has to conform to np.savetxt
        umapCoordsName = "TestUMAPCoords.csv"
    
    # call UMAP
    umap_results = None
    if runUMAP:
        umap_results = utils.SpatialUMAP(
            os.path.join(dataPath, inputName),
            saveCoords,
            os.path.join(savePath, umapCoordsName),
            UMAP_nn,
            UMAP_minDist,
            UMAP_metric
            )




    return



if __name__ == "__main__":
    main()
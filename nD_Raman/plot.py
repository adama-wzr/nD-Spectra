'''
Plotting Sub-Routines

Authors:
    - Andre Adam
    - 
List of Major Updates (date):

    - Sep 4th, 2025 --> project created
'''

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import umap
import umap.plot

plt.rcParams["font.family"] = "Times New Roman"


def plotMeanRaman_sub(nClusters, clusterAvgRaman, Wavelength):
    '''
    Function plotMeanRaman_sub:
    Inputs:
        - (int) number of clusters
        - (ndarray, size (nWave, nClusters)) normalized mean raman spectra
        - (ndarray, size (nWave)) wavelengths for Raman spectroscopy data
    Outputs:
        - none
    
    Function will create a figure plotting each normalized Raman spectra from each cluster
    on a separate subplot.
    '''
    # create subplots
    figK, axs = plt.subplots(nClusters, 1, tight_layout=True)
    # figK.set_dpi(2400)
    figK.set_size_inches(6, 8)
    # make sure all axis have same ticks
    WaveTicks = np.arange(Wavelength[0], Wavelength[-1], (Wavelength[-1] - Wavelength[0])/10)
    maxNormInt=np.max(clusterAvgRaman)
    NormIntTicks = np.arange(0.0, maxNormInt, maxNormInt/10)
    # iterate over clusters
    for n in range(nClusters):
        # plot
        axs[n].plot(Wavelength, clusterAvgRaman[:,n])
        # labels
        axs[n].set_title(f'Cluster {n}')
        axs[n].set_xlabel(r'Wavelength [$\mathrm{\lambda}$]', fontsize=16)
        axs[n].set_ylabel('Normalized Intensity', fontsize=16)
        # ticks
        axs[n].set_xticks(WaveTicks)
        axs[n].set_yticks(NormIntTicks)
        axs[n].yaxis.set_major_formatter(tkr.FormatStrFormatter('%.3f'))
        # limits
        axs[n].set_xlim(WaveTicks[0], WaveTicks[-1])
        axs[n].set_ylim(NormIntTicks[0], 1.1*NormIntTicks[-1])
    
    # save plot
    figK.savefig('k5_1over12_PEO.png')


def plotMeanRaman(nClusters, clusterAvgRaman, Wavelength):
    '''
    Function plotMeanRaman:
    Inputs:
        - (int) number of clusters
        - (ndarray, size (nWave, nClusters)) normalized mean raman spectra
        - (ndarray, size (nWave)) wavelengths for Raman spectroscopy data
    Outputs:
        - none
    
    Function will create a figure plotting all normalized Raman spectra 
    on top of each other.
    '''
    # open figure
    fig, ax = plt.subplots()

    for kN in range(nClusters):
        ax.plot(Wavelength[:], clusterAvgRaman[:,kN], label=kN)

    # waveLengthTicks = np.arange(Wavelength[0], Wavelength[-1], (Wavelength[-1] - Wavelength[0])/10)

    ax.legend(loc='best')
    ax.set_xlabel(r'Wavelength [$\mathrm{\lambda}$]', fontsize=16)
    ax.set_ylabel('Normalized Intensity', fontsize=16)

    # save plot
    fig.savefig('k5_1over12_overlap.png')


def plotMeanRaman_stack(nClusters, clusterAvgRaman, Wavelength):
    '''
    Function plotMeanRaman:
    Inputs:
        - (int) number of clusters
        - (ndarray, size (nWave, nClusters)) normalized mean raman spectra
        - (ndarray, size (nWave)) wavelengths for Raman spectroscopy data
    Outputs:
        - none
    
    Function will create a figure plotting all normalized Raman spectra 
    stacked over each other.
    '''
    # open figure
    figS, ax = plt.subplots()

    # Manually set offset per plot
    offset = 0.01

    for kN in range(nClusters):
        ax.plot(Wavelength[:], clusterAvgRaman[:,kN] + kN*offset, label=kN)

    # waveLengthTicks = np.arange(Wavelength[0], Wavelength[-1], (Wavelength[-1] - Wavelength[0])/10)

    ax.legend(loc='best')
    ax.set_xlabel(r'Wavelength [$\mathrm{\lambda}$]', fontsize=16)
    ax.set_ylabel('Normalized Intensity', fontsize=16)

    # save plot
    figS.savefig('k5_1over12_stack.png')
    
    return


def reconstructLabels(labels):
    '''
    Function reconstructLabels:
    Inputs:
        - (ndarray) mapping labels (flattened)
    Outputs:
        - none
    Function will cast the flat Raman labels to a
    2D shape using C-indexing.
    '''
    labels = np.reshape(labels, [120, 120])

    figC, ax = plt.subplots()
    CSA1 = ax.imshow(labels, cmap='viridis')

    # save plot
    figC.savefig('k5_1over12_cmap.png')

    return

def plotUMAP(my_map, saveUMAP, filename):
    '''
    Function plotUMAP:
    Inputs:
        - (ndarray) umap embedding
        - (bool) true or false for saving plot
        - (string) file name for saving
    Outputs:
        - none
    
    Plots UMAP, may or may not save the plot depending
    on user input.
    '''
    umap.plot.points(my_map)

    return
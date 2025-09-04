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
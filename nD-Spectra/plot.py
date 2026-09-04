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
import matplotlib.patches as mpatches
import umap
import umap.plot
import seaborn as sns

plt.rcParams["font.family"] = "Times New Roman"


def plotMeanRaman_sub(nClusters, clusterAvgRaman, RamanShift, filename):
    '''
    Function plotMeanRaman_sub:
    Inputs:
        - (int) number of clusters
        - (ndarray, size (nWave, nClusters)) normalized mean raman spectra
        - (ndarray, size (nWave)) RamanShifts from Raman spectroscopy data
        - (string) filename
    Outputs:
        - none
    
    Function will create a figure plotting each normalized Raman spectra from each cluster
    on a separate subplot.
    '''
    # create subplots
    figK, axs = plt.subplots(nClusters, 1, tight_layout=True)
    # figK.set_dpi(2400)
    figK.set_size_inches(6, 3*nClusters)
    # make sure all axis have same ticks
    WaveTicks = np.arange(RamanShift[0], RamanShift[-1], (RamanShift[-1] - RamanShift[0])/10)
    maxNormInt=np.max(clusterAvgRaman)
    NormIntTicks = np.arange(0.0, maxNormInt, maxNormInt/10)
    # iterate over clusters
    for n in range(nClusters):
        # plot
        axs[n].plot(RamanShift, clusterAvgRaman[:,n])
        # labels
        axs[n].set_title(f'Cluster {n}')
        axs[n].set_xlabel(r'RamanShift [$\mathrm{cm}^{-1}$]', fontsize=16)
        axs[n].set_ylabel('Normalized Intensity', fontsize=16)
        # ticks
        axs[n].set_xticks(WaveTicks)
        axs[n].set_yticks(NormIntTicks)
        axs[n].yaxis.set_major_formatter(tkr.FormatStrFormatter('%.3f'))
        # limits
        axs[n].set_xlim(WaveTicks[0], WaveTicks[-1])
        axs[n].set_ylim(NormIntTicks[0], 1.1*NormIntTicks[-1])
    
    # save plot
    figK.savefig(filename)

    return


def plotMeanRaman(nClusters, clusterAvgRaman, RamanShift, colors, filename):
    '''
    Function plotMeanRaman:
    Inputs:
        - (int) number of clusters
        - (ndarray, size (nWave, nClusters)) normalized mean raman spectra
        - (ndarray, size (nWave)) RamanShifts for Raman spectroscopy data
        - (cmap) colormap 
        - (string) filename
    Outputs:
        - none
    
    Function will create a figure plotting all normalized Raman spectra 
    on top of each other.
    '''
    # open figure
    fig, ax = plt.subplots(tight_layout=True)

    for kN in range(nClusters):
        ax.plot(RamanShift[:], clusterAvgRaman[:,kN], label=f'Cluster {kN}', color=colors(kN))

    # waveLengthTicks = np.arange(Wavelength[0], Wavelength[-1], (Wavelength[-1] - Wavelength[0])/10)

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    ax.set_xlabel(r'RamanShift [$\mathrm{cm}^{-1}$]', fontsize=16)
    ax.set_ylabel('Normalized Intensity', fontsize=16)

    # save plot
    fig.savefig(filename)

    return


def plotMeanRaman_stack(nClusters, clusterAvgRaman, RamanShift, colors, filename):
    '''
    Function plotMeanRaman:
    Inputs:
        - (int) number of clusters
        - (ndarray, size (nWave, nClusters)) normalized mean raman spectra
        - (ndarray, size (nWave)) RamanShifts for Raman spectroscopy data
        - (cmap) array colors
        - (string) filename
    Outputs:
        - none
    
    Function will create a figure plotting all normalized Raman spectra 
    stacked over each other.
    '''
    # open figure
    figS, ax = plt.subplots(tight_layout=True)

    # Manually set offset per plot
    offset = 0.01

    for kN in range(nClusters):
        ax.plot(RamanShift[:], clusterAvgRaman[:,kN] + kN*offset, label=f'Cluster {kN}', color=colors(kN))

    # waveLengthTicks = np.arange(Wavelength[0], Wavelength[-1], (Wavelength[-1] - Wavelength[0])/10)

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    ax.set_xlabel(r'RamanShift [$\mathrm{cm}^{-1}$]', fontsize=16)
    ax.set_ylabel('Normalized Intensity', fontsize=16)

    # save plot
    figS.savefig(filename)
    
    return


def reconstructLabels(labels, colors, nK, InName, OutName):
    '''
    Function reconstructLabels:
    Inputs:
        - (ndarray) mapping labels (flattened)
        - (ndarray) color pallette from plt.cm.get_cmap
        - (int) number of clusters
        - (string) InName -> input filename will be plotted as title
        - (string) OutName -> output filename will save file name
    Outputs:
        - none
    Function will cast the flat Raman labels to a
    2D shape using C-indexing.
    '''

    # get coverage percent
    labelsCt = np.zeros(nK)

    for i in range(len(labels)):
        labelsCt[labels[i]] += 1

    # Cluster Coverage Percentages
    print(InName)
    for k in range(nK):
        print(f'Cluster {k} = {float(labelsCt[k]/len(labels))*100}')

    labels = np.reshape(labels, [120, 120])

    values = np.unique(labels.ravel())

    figC, ax = plt.subplots(tight_layout=True)
    # CSA1 = ax.imshow(labels, cmap=colors, interpolation='none')
    ax.pcolor(labels, cmap=colors, vmin=0, vmax=nK)

    # colors = [CSA1.cmap(CSA1.norm(value)) for value in values]

    patches = [ mpatches.Patch(color=colors(values[i]), label="Cluster {l}".format(l=values[i]) ) for i in range(len(values)) ]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=16)
    ax.set_xticks([])
    ax.set_xticklabels([])

    ax.set_yticks([])
    ax.set_yticklabels([])

    ax.set_title(InName)

    # save plot
    figC.savefig(OutName)

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
    umapPlot = umap.plot.points(my_map)

    if saveUMAP:
        fig = umapPlot.get_figure()
        fig.savefig(filename)

    return

def plotUMAP_Clusters(coords, labels, colors, filename):
    '''
    Function plotUMAP_Clusters:
    Inputs:
        - umap coordinates
        - ndarray with cluster labels
        - colors array (plt.cm.get_cmap)
        - save file name
    Outputs:
        - none
    Function will cast the cluster labels back onto the umap
    and plot it again.
    '''

    # get labels
    unique_labels = np.unique(labels)

    plt.figure(tight_layout=True)

    for i,label in enumerate(unique_labels):
        # filter subsets
        indices = np.where(labels == label)
        x = coords[indices, 0]
        y = coords[indices, 1]

        plt.scatter(x, y, color=colors(i), marker='.', label=f'Cluster {label}')
    plt.xticks([],[])
    plt.yticks([],[])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

    plt.savefig(filename)

    return

def plotKMeansOpt(kRange, ScoreList, filename):
    '''
    Function plotKMeansOpt:
    Inputs:
        - (ndarray) range of K's
        - (list) scores
        - (string) filename + path
    Outputs:
        - None
    Function will plot the k-Means optimization,
    save the figure, and save the values onto a .csv
    file.
    '''
    array = np.zeros((len(kRange), 2))
    for i in range(len(kRange)):
        array[i,0] = kRange[i]
        array[i,1] = ScoreList[i]
    
    figSNS, axSNS = plt.subplots()
    sns.lineplot(x = kRange, y = ScoreList, ax=axSNS)

    # filename
    figureName = str(filename) + '.png'
    csvName = str(filename) + '.csv'

    figSNS.savefig(figureName)
    np.savetxt(csvName, array, delimiter=',', header='k,k-Score')
    
    return
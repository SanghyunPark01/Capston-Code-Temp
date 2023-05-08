#path
path = "./LowAlphaArray.npy"

#--------------------------------------------------------------------------
import torch
import cv2
import math
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
import matplotlib.pyplot as plt
#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D

def main() : 
    npLowAlphaGrid = np.load(path)
    #print(npLowDensityGrid.shape[0])

    npA_idxZ, npA_idxY, npA_idxX = np.where(npLowAlphaGrid == 0)

    # print(npA_idxZ.shape)
    # print(npA_idxY.shape)
    # print(npA_idxX.shape)
    print("indexing start")

    npaLowAlphaArr = np.zeros((npA_idxZ.shape[0],3))
    print("Number of Points : ",npaLowAlphaArr.shape[0])

    for idx in range(0,npA_idxZ.shape[0]):
        npaLowAlphaArr[idx,2] = npA_idxX[idx]
        npaLowAlphaArr[idx,1] = npA_idxY[idx]
        npaLowAlphaArr[idx,0] = npA_idxZ[idx]

    print("idexing complete")

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(npaLowDensityArr[:10000,0], npaLowDensityArr[:10000,1], npaLowDensityArr[:10000,2], s=1)
    # ax.view_init(azim=200)
    # plt.show()
    print("clustering start")
    print("On going...")
    model = DBSCAN(eps=2.5, min_samples=5)
    pred = model.fit_predict(npaLowAlphaArr)
    print("clustering complete")
    labels = model.labels_
    print("clustering result : ", model.labels_)


    nNclusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("number of cluster: ",nNclusters)

    nNoiseIdx = nNclusters
    npClusteringIDX = np.zeros(nNclusters+1) # clustering idx : 0~(nNclusters-1), noise idx : nNclusters
    
    print("Analyzing data...")
    nMaxSizeIndex = nNoiseIdx
    for clusterLabel in labels:
        if clusterLabel == -1:
            npClusteringIDX[nNoiseIdx] += 1
        npClusteringIDX[clusterLabel] += 1
        if (npClusteringIDX[nMaxSizeIndex] < npClusteringIDX[clusterLabel] and clusterLabel != nNoiseIdx):
            nMaxSizeIndex = clusterLabel

    nMaxIndexSize = npClusteringIDX[nMaxSizeIndex]
    print("    < information >")
    print("Big size clustering label : ",nMaxSizeIndex)
    print("ㄴNumber this label : ",nMaxIndexSize)

    # Find Point
    nMax_X = 0
    nMax_Y = 0
    nMax_Z = 0
    print("Find projection origin point...")
    for idx in range(0, npaLowAlphaArr.shape[0]):
        if labels[idx] == nMaxSizeIndex:
            #X
            nMax_X += npaLowAlphaArr[idx,2]/nMaxIndexSize
            #Y
            nMax_Y += npaLowAlphaArr[idx,1]/nMaxIndexSize
            #Z
            nMax_Z += npaLowAlphaArr[idx,0]/nMaxIndexSize
    print("Find!!")
    print("origin point : (x,y,z) = (", nMax_X, nMax_Y, nMax_Z,")")

    #visualize
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(npaLowAlphaArr[:,2], npaLowAlphaArr[:,1], npaLowAlphaArr[:,0], c=model.labels_, s=1)
    ax.view_init(azim=-100,elev=-34)
    plt.show()

    


if __name__ == '__main__':
    main()
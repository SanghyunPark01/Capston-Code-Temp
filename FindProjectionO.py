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
    print(npaLowAlphaArr.shape)

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

    model = DBSCAN(eps=2.5, min_samples=5)
    #model.fit_predict(npaLowDensityArr)
    pred = model.fit_predict(npaLowAlphaArr)
    print("clustering complete")

    print("number of cluster found: {}".format(len(set(model.labels_))))
    print('cluster for each point: ', model.labels_)
    print(model.components_.shape)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(npaLowAlphaArr[:,2], npaLowAlphaArr[:,1], npaLowAlphaArr[:,0], c=model.labels_, s=1)
    ax.view_init(azim=-100,elev=-34)
    plt.show()

    


if __name__ == '__main__':
    main()
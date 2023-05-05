rgb_imageDirectory_path = "./rgb_256" 

import numpy as np
import cv2

def main():
    nColorImageSize = 256

    npAlphaArray = np.zeros((256,256,256))

    for i in range(0, nColorImageSize) :
        sRGBPath = rgb_imageDirectory_path + "/{0:>04d}_256x256.png".format(i)
        cvRGBImg = cv2.imread(sRGBPath, cv2.IMREAD_UNCHANGED) #IMREAD_COLOR IMREAD_UNCHANGED
        npAlpha = cvRGBImg[:,:,3:]
        npAlpha = npAlpha.squeeze()
        npAlphaArray[nColorImageSize-i-1] = npAlpha
    print("indexing complete")
    np.save('./LowAlphaArray', npAlphaArray)
    print("Save Complete!")
    
    
    # sRGBPath = rgb_imageDirectory_path + "/{0:>04d}_256x256.png".format(205)
    # cvRGBImg = cv2.imread(sRGBPath, cv2.IMREAD_UNCHANGED)
    # print(cvRGBImg.shape)
    # cvAlpha = cvRGBImg[:,:,3:]
    # test = cvAlpha.squeeze()
    # print(test.shape)
    # npA_idxY, npA_idxX = np.where(test == 0)
    # print(npA_idxX)
    # print(npA_idxY)

    #print(cvRGBImg[120,120])
    #cv2.imshow("a",cvRGBImg)
    #cv2.waitKey()
if __name__ == '__main__':
    main()
'''
----------------------------------------------------------------------------
          < InstantNGP-result to Torch Converter for NeRF-GAN >

This Code is that convert InstantNGP's result of rgb & dense to Torch.
It is for NeRF-GAN(On going Porject).

Implemented by Sanghyun Park.

Github : https://github.com/SanghyunPark01 (SanghyunPark01)
E-Mail : pash0302@naver.com / pash0302@gmail.com
----------------------------------------------------------------------------
'''

#path
dense_image_path = "./density_256.png"
rgb_imageDirectory_path = "./rgb_256" 
#size
dense_image_size = 256
color_image_size = 256

#--------------------------------------------------------------------------
import torch
import cv2
import math
import numpy as np

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

def main() : 
    sRGB_DirectoryPath = rgb_imageDirectory_path
    sDensePath = dense_image_path
    nDenseImageSize = dense_image_size
    nColorImageSize = color_image_size

    cvDenseImage = cv2.imread(sDensePath, cv2.IMREAD_UNCHANGED)
    #cv2.imshow('dense_image',cvDenseImage)
    #cv2.waitKey()
    
    tTensorShape = (nColorImageSize,nColorImageSize,nColorImageSize,4) # index : (z,y,x) => value : (b, g, r, density)
    tINGPInfo = torch.zeros(tTensorShape).to(device)

    #tDenseZeroInfo = torch.zeros(nColorImageSize,nColorImageSize,nColorImageSize).to(device)

    npaLowDensityArr = np.empty((0,3),int)

    #print(tINGPInfo.shape)
    #print(tINGPInfo)
    #print(tDenseZeroInfo.shape)
    #print(tDenseZeroInfo)

    nDense_row_O = 0
    nDense_col_O = 0
    for i in range(0, nColorImageSize) :
        sRGBPath = sRGB_DirectoryPath + "/{0:>04d}_256x256.png".format(i)
        cvRGBImg = cv2.imread(sRGBPath, cv2.IMREAD_COLOR) #IMREAD_COLOR IMREAD_UNCHANGED
        # cv2.imshow("rgb",cvRGBImg)
        # cv2.waitKey(10)
        for j in range(0, nColorImageSize) : 
            for k in range(0, nColorImageSize) :
            # (255 - i), j, k => z(image sequence), y(row), x(col)
                nDense_row_temp = nDense_row_O + j
                nDense_col_temp = nDense_col_O + k
                nDense_value = cvDenseImage[nDense_row_temp, nDense_col_temp]
                #print("image : ",i,", row : ",nDense_row_temp,", col : ",nDense_col_temp)

                nDenseZero_temp = 0
                if nDense_value < 1:
                    nDenseZero_temp = 1

                nIndex_Z = 255 - i
                nIndex_Y = j
                nIndex_X = k

                b, g, r = cvRGBImg[j,k]

                tINGPInfo[nIndex_Z,nIndex_Y,nIndex_X] = torch.tensor([b, g, r, nDense_value]).to(device)
                if nDenseZero_temp > 0 :
                    npaLowDensityArr = np.append(npaLowDensityArr,np.array([[nIndex_X,nIndex_Y,nIndex_Z]]), axis = 0)

        nDense_col_O = nDense_col_O + nColorImageSize
        if nDense_col_O >= math.sqrt(nDenseImageSize) * nDenseImageSize :
            nDense_row_O += nColorImageSize
            nDense_col_O = 0
        
        print("Progress Rate :",(i/nColorImageSize)*100,"%")
        
    #Result - tINGPInfo
    print("----------------------------")
    print("Convert Complete!")
    print(tINGPInfo.shape)
    
    #Save Low Density [x,y,z] array
    np.save('./LowDensityArray', npaLowDensityArr)
    print("Save Complete!")

if __name__ == '__main__':
    main()

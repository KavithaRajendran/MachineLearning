# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:21:56 2017
@author: Kavitha Rajendran
"""

import numpy as np
import cv2
import os
import sys
import glob

#number of clusters as command line argument
if(len(sys.argv)<2):
    print("Please mention number of clusters")
    sys.exit(0)
K = int(sys.argv[1])

#get all list of image files
listOfImgs = glob.glob(r'./inputImages/*.jpg')
print("listOfImgs:",listOfImgs)
count=0;
for i in listOfImgs:
    #reading input image file
    sourceImage = cv2.imread(i)
    img = sourceImage.reshape((-1,3))
    img = np.float32(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #apply kmeans()
    compactness,labels,centers=cv2.kmeans(img,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    result = centers[labels.flatten()]
    #reshape to original shape
    result = result.reshape((sourceImage.shape))
    resultDirectory="clusteredImages"
    if not os.path.exists(resultDirectory):
        os.makedirs(resultDirectory)
    resultFile=resultDirectory+"/image"+str(count)+".jpg"
    #writing output image file to the output folder
    cv2.imwrite(resultFile,result)
    count+=1

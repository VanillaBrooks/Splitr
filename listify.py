import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pathlib import Path

def listify(receipt):
    img = cv.imread(receipt)
#    cv.imshow('Original',img)

    ## (1) read
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#    cv.imshow('Grayscale',gray)

    ## (2) threshold
    th, threshed = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
#    cv.imshow('Threshold1',threshed)

    threshed = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,13,13)
#    cv.imshow('Threshold3',threshed)


    ## (3) minAreaRect on the nozeros
    pts = cv.findNonZero(threshed)
    ret = cv.minAreaRect(pts)

    (cx,cy), (w,h), ang = ret
    if w>h:
        w,h = h,w
        ang += 90

    ## (4) Find rotated matrix, do rotation
    M = cv.getRotationMatrix2D((cx,cy), ang, 1.0)
    rotated = cv.warpAffine(threshed, M, (img.shape[1], img.shape[0]))

#    cv.imshow('Rotated',rotated)
    ## (5) find and draw the upper and lower boundary of each lines
    hist = cv.reduce(rotated,1, cv.REDUCE_AVG).reshape(-1)

    th = 2
    H,W = img.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

    rotated = cv.bitwise_not(rotated)

#    for y in uppers:
#        cv.line(rotated, (0,y), (W, y), (255,0,0), 1)
#    for y in lowers:
#        cv.line(rotated, (0,y), (W, y), (0,255,0), 1)

#    cv.imshow("Result", rotated)

    cropcounter = 1
    imgwidth = rotated.shape[1]
    x,y = rotated.shape
    final = rotated.reshape(x,y,1)

    listified = np.array()
    for u,l in list(zip(uppers,lowers)):
            section = final[u:l, 0:imgwidth]
            listified.append(section)
            
            #filename = "receiptline_" + str(cropcounter) + ".jpg"
            #result = Path('Results/' + filename)
            #cv.imwrite(str(result), section)
            #cropcounter+=1;
            #print (section.shape)
    return listified

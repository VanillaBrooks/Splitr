import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

SAVE_PATH = r'Results'
CLEAR_OLD_DATA = True

def listify(receipt):
	input_file_name = receipt.split('.')[0] # get the name of the file
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

	# if we want fresh folder each time delete the previous one
	if CLEAR_OLD_DATA:
		files = os.listdir(SAVE_PATH)
		for f in files:
			os.remove(os.path.join(SAVE_PATH,f))

	# create the reciepts directory if it does not exist
	if not os.path.exists(SAVE_PATH):
		os.mkdir(SAVE_PATH)

	listified = []
	for u,l in list(zip(uppers,lowers)):
			section = final[u:l, 0:imgwidth]
			listified.append(section)

			filename = input_file_name + "_receiptline_" + str(cropcounter) + ".jpg"
			result = os.path.join(SAVE_PATH, filename)
			cv.imwrite(str(result), section)
			cropcounter+=1;
			print (section.shape)
	return listified

if __name__ == '__main__':
	listify('receipt2.jpg')

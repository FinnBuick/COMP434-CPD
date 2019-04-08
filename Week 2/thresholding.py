import cv2
import numpy as np

img = cv2.imread('Week 2\\bookpage.jpg')

#grayscale the img to see if that makes the text more readable
grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#adaptive thresholding attempts to vary the threshold to account for the curve in page
th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

retval, threshold = cv2.threshold(grayscaled, 12, 225, cv2.THRESH_BINARY)
cv2.imshow('original', img)
cv2.imshow('Adaptive Threshold', th)
cv2.waitKey(0)
cv2.destroyAllWindows()

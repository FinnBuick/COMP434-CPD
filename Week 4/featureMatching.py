import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('Week 4\\opencv-feature-matching-template.jpg', 0)
img2 = cv2.imread('Week 4\\opencv-feature-matching-image.jpg', 0)

orb = cv2.ORB_create()

kp1, desc1 = orb.detectAndCompute(img1, None)
kp2, desc2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(desc1, desc2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1,kp1,img2,kp2, matches[:30], None, flags=2)
plt.imshow(img3)
plt.show()

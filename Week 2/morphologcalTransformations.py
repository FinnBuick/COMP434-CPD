import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Convert frames to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV hue sat value
    lower_red = np.array([150,120,50])
    upper_red = np.array([200,255,255])

    # Create a mask for pixel within the upper and lower bounds
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # And the frame with the mask to show color only where the color is in the range
    res = cv2.bitwise_and(frame,frame, mask = mask)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    dilation = cv2.dilate(mask,kernel,iterations = 1)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow('Original',frame)
    # cv2.imshow('Mask',mask)
    # cv2.imshow('Erosion',erosion)
    # cv2.imshow('Dilation',dilation)
    cv2.imshow('Opening',opening)
    cv2.imshow('Closing',closing)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()

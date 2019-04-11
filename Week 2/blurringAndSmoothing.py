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

    kernel = np.ones((15,15),np.float32)/225

    smoothed = cv2.filter2D(res,-1,kernel)

    blur = cv2.GaussianBlur(res,(15,15),0)

    median = cv2.medianBlur(res,15)

    bilateral = cv2.bilateralFilter(res,15,75,75)


    # Display all three stages
    cv2.imshow('frame', frame)
    #cv2.imshow('mask', mask)
    cv2.imshow('result', res)
    #cv2.imshow('Averaging',smoothed)
    #cv2.imshow('Gaussian Blurring',blur)
    #cv2.imshow('Median Blur',median)
    cv2.imshow('bilateral Blur',bilateral)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# def nothing(x):
#     pass
#
# cv2.namedWindow('Trackbars')
#
# cv2.createTrackbar('L - H', 'Trackbars', 0, 179, nothing)
# cv2.createTrackbar('L - S', 'Trackbars', 0, 255, nothing)
# cv2.createTrackbar('L - V', 'Trackbars', 0, 255, nothing)
# cv2.createTrackbar('U - H', 'Trackbars', 0, 179, nothing)
# cv2.createTrackbar('U - S', 'Trackbars', 0, 255, nothing)
# cv2.createTrackbar('U - V', 'Trackbars', 0, 255, nothing)

while(True):
    ret, frame = cap.read()
    # Convert from BGR to Gray
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # l_h = cv2.getTrackbarPos('L - H', "Trackbars")
    # l_s = cv2.getTrackbarPos('L - S', "Trackbars")
    # l_v = cv2.getTrackbarPos('L - V', "Trackbars")
    # u_h = cv2.getTrackbarPos('U - H', "Trackbars")
    # u_s = cv2.getTrackbarPos('U - S', "Trackbars")
    # u_v = cv2.getTrackbarPos('U - V', "Trackbars")
    #
    #
    # # HSV hue sat value
    # lower_red = np.array([l_h,l_s,l_v])
    # upper_red = np.array([u_h,u_s,u_v])

    lower_red = np.array([0,61,112])
    upper_red = np.array([36,180,255])

    # Create a mask for pixel within the upper and lower bounds
    mask = cv2.inRange(hsv, lower_red, upper_red)

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)

    mask = cv2.GaussianBlur(mask, (5,5), 100)

    frame2, contours, hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    res = cv2.bitwise_and(frame,frame, mask = mask)

    # draw contours
    cv2.drawContours(res, contours, -1, (0,0,255), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('res', res)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('Week 4\\haarcascade_frontalface_default.xml')

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

def colorFilter(hsv):
    lower_red = np.array([0,61,112])
    upper_red = np.array([36,180,255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    return mask


while True:
    ret, frame = cap.read()
    # Convert from BGR to Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Detect face and remove it from the image
    # faces = face_cascade.detectMultiScale(gray, 1.05, 5)
    #
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(hsv, (x,y-40), (x+w, y+h+40), (0,0,0), -1)

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



    # Create a mask for pixel within the upper and lower bounds
    mask = colorFilter(hsv)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.GaussianBlur(mask, (7,7), 100)
    res = cv2.bitwise_and(frame,frame, mask = mask)

    frame2, contours, hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        # draw contours in blue
        cv2.drawContours(res, contours, -1, (255,255,255), 2)

        # Find the biggest area
        max_contour = max(contours, key = cv2.contourArea)

        # Draw bounding rect
        x,y,w,h = cv2.boundingRect(max_contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        # Find center of the rectangle
        center = (x + w/2, y + h/2)

        # Find the convex hull

        hull = cv2.convexHull(max_contour, returnPoints = False)
        defects = cv2.convexityDefects(max_contour, hull)
        count = 0
        if defects is not None:
            highest_point = (0,0)
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]


                if d > 3000:
                    count+=1
                    start = tuple(max_contour[s][0])
                    end = tuple(max_contour[e][0])
                    far = tuple(max_contour[f][0])
                    cv2.line(frame,start,end,[0,255,0],2)
                    cv2.circle(frame,far,5,[0,0,255],-1)


        top_point = tuple(max_contour[max_contour[:, :, 1].argmin()][0])

        if (center[1] - top_point[1]) > 200:
            count+=1

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(count),(10,30), font, 1,(255,255,255),2,cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.imshow('res', res)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

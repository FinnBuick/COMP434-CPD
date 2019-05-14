import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


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

def histMasking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject([hsv], [0,1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(backproj, -1, disc, backproj)
    ret, thresh = cv2.threshold(backproj, 150, 255, cv2.THRESH_BINARY)


    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh,kernel,iterations = 1)
    thresh = cv2.GaussianBlur(thresh, (7,7), 100)

    thresh = cv2.merge((thresh, thresh, thresh))
    return cv2.bitwise_and(frame, thresh)

def defectAngle(start, end, far):
    """Calculate angle of defect using cosine rule"""
    a = abs(end[0] - start[0]) + abs(end[1] - start[1])
    b = abs(start[0] - far[0]) +  abs(start[1] - far[1])
    c = abs(end[0] - far[0]) + abs(end[1] - far[1])
    angle = math.acos((b**2 + c**2 - a**2) / (2*b*c))
    return angle

def countFingers(frame, contour):
    """Takes a countour and performs finger detection on it by counting the
    number of convexity defects"""
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    center = (x + w/2, y + h/2)

    hull = cv2.convexHull(contour, returnPoints = False)
    defects = cv2.convexityDefects(contour, hull)
    count = 0
    if defects is not None:
        highest_point = (0,0)
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            angle = defectAngle(start, end, far)
            cv2.line(frame,start,end,[255,0,0],2)
            cv2.circle(frame,far,5,[0,0,255],-1)

            if d > 5000 and angle <= math.pi/2:
                count+=1

        """Use the distance between the highest point of the contour and the
        center of the bounding rectangle to determine if a single finger is
        being held up"""
        top_point = tuple(contour[contour[:, :, 1].argmin()][0])
        if (center[1] - top_point[1]) > 200:
            count+=1

        return True, count
    return False, 0

def skinHistogram(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rows, columns, _ = hsv.shape

    num_rects = 6

    # Draw the rectangles to indicate hand placement
    top_left_x = np.array([
    12 * rows / 20, 12 * rows / 20, 12 * rows / 20,
    15 * rows / 20, 15 * rows / 20, 15 * rows / 20], dtype=np.uint32)

    top_left_y = np.array([
    4 * columns / 20, 6 * columns / 20, 8 * columns / 20,
    4 * columns / 20, 6 * columns / 20, 8 * columns / 20], dtype=np.uint32)

    bottom_right_x = top_left_x + 10
    bottom_right_y = top_left_y + 10

    for i in range(num_rects):
        cv2.rectangle(frame, (top_left_x[i], top_left_y[i]),
        (bottom_right_x[i], bottom_right_y[i]), (255,255,255), 1)


    roi = np.zeros([60, 10, 3], dtype=hsv.dtype)

    for i in range(num_rects):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv[top_left_x[i]:top_left_x[i] + 10, top_left_y[i]:top_left_y[i] + 10]

    skin_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])

    return cv2.normalize(skin_hist, skin_hist, 0, 255, cv2.NORM_MINMAX)

def main():
    hist_created = False

    while True:
        ret, frame = cap.read()
        masked_img = np.zeros(frame.shape, dtype=frame.dtype)
        # Convert from BGR to Gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect face and remove it from the image
        faces = face_cascade.detectMultiScale(gray, 1.05, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(hsv, (x,y-40), (x+w, y+h+40), (0,0,0), -1)

        # l_h = cv2.getTrackbarPos('L - H', "Trackbars")
        # l_s = cv2.getTrackbarPos('L - S', "Trackbars")
        # l_v = cv2.getTrackbarPos('L - V', "Trackbars")
        # u_h = cv2.getTrackbarPos('U - H', "Trackbars")
        # u_s = cv2.getTrackbarPos('U - S', "Trackbars")
        # u_v = cv2.getTrackbarPos('U - V', "Trackbars")
        #
        # #
        # # # HSV hue sat value
        # lower_red = np.array([l_h,l_s,l_v])
        # upper_red = np.array([u_h,u_s,u_v])

        if cv2.waitKey(1) & 0xFF == ord('s'):
            hist_created = True
            hist = skinHistogram(frame)

        if hist_created:
            # Create a mask for pixel within the upper and lower bounds
            masked_img = histMasking(frame, hist)
            gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
            _, contours, hierachy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # draw contours in blue
                cv2.drawContours(frame, contours, -1, (0,255,0), 2)

                # Find the biggest area
                max_contour = max(contours, key = cv2.contourArea)

                fingersPresent, count = countFingers(frame, max_contour)
                if fingersPresent:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, str(count),(10,30), font, 1,(0,0,0),2,cv2.LINE_AA)


        cv2.imshow('frame', frame)
        # cv2.imshow('res', res)
        cv2.imshow('mask', masked_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

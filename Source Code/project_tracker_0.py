import numpy as np
import cv2
import time

# video source
cap = cv2.VideoCapture('video/fixed/MOT16-03-raw.mp4')


# Specify the tracker type
print("Please select tracker type")
print("0:'Meanshift', 1:'Camshift'")
option1 = int(input())


# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

# take first frame of the video
success,frame = cap.read()

# setup initial location of window
x,y,w,h = 0, 0, 0, 0

#select tracking bounding box


print("Input bounding box value x")
x = int(input())
print("Input bounding box value y")
y = int(input())
print("Input bounding box value w")
w = int(input())
print("Input bounding box value h")
h = int(input())


#x,y,w,h = cv2.selectROI(frame)
print("Selected bounding boxes [("+str(x)+", "+ str(y)+", "+ str(w)+", "+ str(h)+ ")]")
track_window = (x,y,w,h)

# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while True:
    success, frame = cap.read()

    if not success:
        break

    # time when we finish processing for this frame
    new_frame_time = time.time()

    # Calculating the fps
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)


    if option1 == 0:
        # apply meanshift to get the new location
        success, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)

        # Display tracker type on frame
        cv2.putText(frame, "Meanshift Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    elif option1 == 1:
        # apply camshift to get the new location
        success, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image
        pts = cv2.boxPoints(success)
        pts = np.int0(pts)
        frame = cv2.polylines(frame, [pts], True, 255, 2)

        # Display tracker type on frame
        cv2.putText(frame, "Camshift Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    else:
        print("invalid tracker type")


    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    #show frame
    cv2.imshow('Meanshift/Camshift',frame)


    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break



cv2.destroyAllWindows()
cap.release()
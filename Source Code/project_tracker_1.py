from __future__ import print_function
import sys
import cv2
import time
from random import randint

# video source
cap = cv2.VideoCapture('video/moving/MOT16-07-raw.mp4')

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker

# Number of Tracking Items need to track
print("Please input no. of items need to track")
option1 = int(input())
print("No. of tracking items : " + str(option1))

# Specify the tracker type
print("Please select tracker type")
print("0:'BOOSTING', 1:'MIL', 2:'KCF', 3:'TLD', 4:'MEDIANFLOW', 5:'GOTURN', 6:'MOSSE', 7:'CSRT'")
option2 = int(input())
print("Tracker type : " + str(trackerTypes[option2]))
trackerType = trackerTypes[option2]

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0


# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
    print('Failed to read video')
    sys.exit(1)

## Select boxes
bboxes = []
colors = []

# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
# So we will call this function in a loop till we are done selecting all objects


i=0
while True and i < option1:
    # draw bounding boxes over objects
    # selectROI's default behaviour is to draw box starting from the center
    # when fromCenter is set to false, you can draw box starting from top left corner

    # input coordinate by keyboard
    print("Object:"+str(i))
    print("Input bounding box value x")
    x = input()
    print("Input bounding box value y")
    y = input()
    print("Input bounding box value w")
    w = input()
    print("Input bounding box value h")
    h = input()
    bbox = (int(x), int(y), int(w), int(h))

    # input coordinate by roi selector
    #bbox = cv2.selectROI('MultiTracker', frame)


    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))


    '''
    # for input coordinate by roi selector only
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    '''

    print('\n')
    i+=1

print('Selected bounding boxes {}'.format(bboxes))

# Create MultiTracker object

multiTracker = cv2.MultiTracker.create()

# Initialize MultiTracker
for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)

# Process video and track objects
while cap.isOpened():
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

    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)

    # draw tracked objects
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.putText(frame, str(i), (int(newbox[0]), int(newbox[1])), cv2.FONT_HERSHEY_PLAIN, 3, colors[i], 5)
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    # Display tracker type on frame
    cv2.putText(frame, trackerType + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display no. of tracking items
    cv2.putText(frame, "No. of tracking items : " + str(i+1), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # show frame
    cv2.imshow('MultiTracker', frame)

    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break
from __future__ import print_function
import sys
import cv2
import time
from random import randint

# video source
cap = cv2.VideoCapture('video/fixed/MOT16-03-raw.mp4')

# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
    print('Failed to read video')
    sys.exit(1)

bbox = cv2.selectROI('selectROI', frame)
print(bbox)
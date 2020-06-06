# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
from imutils.video import FileVideoStream
import numpy as np
import argparse
import time
import cv2
import dlib
import imutils

from drawables import get
from drawables import getDrawableNames
from my_utils import overlay_transparent
from my_utils import eye_aspect_ratio

fileStream = False

alpha = 0.4
confidence = 0.5
drawDetectionInfo = False

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=False,
	help="path to Caffe 'deploy' prototxt file", default="models/face_detection/deploy.prototxt.txt")
ap.add_argument("-m", "--model", required=False,
	help="path to Caffe pre-trained model", default="models/face_detection/res10_300x300_ssd_iter_140000.caffemodel")
ap.add_argument("-c", "--confidence", type=float, default=confidence,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--shape-predictor", help="Path to facial landmark predictor",
	default="models/facial_landmark_prediction/shape_predictor_68_face_landmarks.dat")
ap.add_argument("-v", "--video", type=str, default="video/test.mp4",
	help="Path to video file")
args = vars(ap.parse_args())

# Load detectors
print("[INFO] loading model...")
# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Cache list of drawable names
drawables = getDrawableNames()

# Variables for blink detection.
# Define two constants, one for the eye aspect ratio to indicate a blink
# and a second constant for the number of consecutive frames
# the eye must be below the threshold
EYE_AR_THRESHOLD = 0.16
EYE_AR_CONSEC_FRAMES = 1

# Initialize the frame counters and the total number of blinks
frame_counter = 0
total_blinks = 0
ear = 0

# Change drawable based on blink count
drawableIndex = 0

# Indexes for facial landmarks for left and right eye respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")

if fileStream:
	# Try video stream from file
	vc = FileVideoStream(args["video"]).start()
	time.sleep(1.0)
else:
	# vs = VideoStream(src=0).start()
	vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	# Forcing resolution increase leads to decrease in frame rate and quality loss
	vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	time.sleep(1.0)

# Video processing main loop
while True:
	if fileStream and not vc.more():
		break
	if fileStream:
		frame = vc.read()
		frame = imutils.resize(frame, width=1280, height=720)
	else:
		ret, frame = vc.read()
		# frame = imutils.resize(frame, width=450)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces
	rects = detector(gray, 0)

	# Loop over face detections
	for rect in rects:
		
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract left and right eye coordinates
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# Average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2			

		# Check if minimum number of consecutive frames has passed since EAR reached threshold
		if ear < EYE_AR_THRESHOLD:
			frame_counter += 1
		else:
			# A blink is detected here
			if frame_counter >= EYE_AR_CONSEC_FRAMES:
				total_blinks += 1
				drawableIndex += 1
				if drawableIndex >= len(drawables):
					drawableIndex = 0
			
			# Reset counter
			frame_counter = 0

		# Write drawable onto frame
		# Convert rect to a tuple of (startX, startY, endX, endY) as required by drawables.py 'get' function
		startX = rect.left()
		startY = rect.top()
		endX = rect.right()
		endY = rect.bottom()
		bounding_box = (startX, startY, endX, endY)
		
		# DEBUGGING
		bounding_box_width = endX - startX
		# END DEBUGGING
		
		(overlay, x, y) = get(drawables[drawableIndex], bounding_box)
		if drawDetectionInfo:
			cv2.putText(frame, "Face width: {}".format(bounding_box_width), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 255, 0), 2)
		if x > 0 and y > 0:
			frame = overlay_transparent(frame, overlay, x, y)

	# Display number of blinks
	if drawDetectionInfo:
		cv2.putText(frame, "Blinks: {}".format(total_blinks), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)

	# Show the output frame
	cv2.imshow("Blink detection", frame)

	# If specified key is pressed while focus is on the window, end the loop
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
		

# do a bit of cleanup
vc.release()
cv2.destroyAllWindows()
# vs.stop()
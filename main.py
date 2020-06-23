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
from playsound import playsound
from threading import Timer
import time
import os
import math

from drawables import get
from drawables import getDrawableNames
from drawables import getCategoryNames
from drawables import getByCategory
from drawables import getCategoryItemCount
from my_utils import overlay_transparent
from my_utils import eye_aspect_ratio
from my_utils import rotate_box
from my_utils import law_of_cosines_three_known_sides
from my_utils import default_EAR_threshold
from scipy.spatial import distance as dist

portrait_mode = False
SCREEN_WIDTH_LANDSCAPE = 1920
SCREEN_HEIGHT_LANDSCAPE = 1080
SCREEN_WIDTH_PORTRAIT = SCREEN_HEIGHT_LANDSCAPE
SCREEN_HEIGHT_PORTRAIT = SCREEN_WIDTH_LANDSCAPE
fileStream = False

confidence = 0.5
drawDetectionInfo = True

# construct the argument parse and parse the arguments if any
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="video/test.mp4",
	help="Path to video file")
args = vars(ap.parse_args())

# Load detectors
print("[INFO] loading model...")
# net = cv2.dnn.readNetFromCaffe('models/face_detection/deploy.prototxt.txt, 'models/face_detection/res10_300x300_ssd_iter_140000.caffemodel')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/facial_landmark_prediction/shape_predictor_68_face_landmarks.dat')

# Create screenshots folders if not yet created
screenshots_folder_name = 'screenshots'
if not os.path.isdir(screenshots_folder_name):
	os.mkdir(screenshots_folder_name)


# Cache list of drawable names
drawables = getDrawableNames()
categories = getCategoryNames()
current_category = categories[0]
current_category_index = 0
category_item_count = getCategoryItemCount(current_category)

# Change drawable based on blink count
drawableIndex = 0

def nextCategory():
	global current_category_index, current_category, category_item_count, drawableIndex
	current_category_index += 1
	if current_category_index >= len(categories):
		current_category_index = 0
	current_category = categories[current_category_index]
	category_item_count = getCategoryItemCount(current_category)
	drawableIndex = 0
	

def prevCategory():
	global current_category_index, current_category, category_item_count, drawableIndex
	current_category_index -= 1
	if current_category_index < 0:
		current_category_index = len(categories) - 1
	current_category = categories[current_category_index]
	category_item_count = getCategoryItemCount(current_category)
	drawableIndex = 0

# Variables for blink detection.
# Define two constants, one for the eye aspect ratio to indicate a blink
# and a second constant for the number of consecutive frames
# the eye must be below the threshold
EYE_AR_THRESHOLD = default_EAR_threshold
EYE_AR_CONSEC_FRAMES = 1

# Initialize the frame counters and the total number of blinks
frame_counter = 0
total_blinks = 0
ear = 0
currently_closed_eye = None # None, 'left', 'right', 'both'

# For doing "screenshot" after a delay, saving camera frame to disk
executing_screenshot = False
screenshot_delay = 3 # In milliseconds
screenshot_thread = None
elapsed_time = 0

def countdown():
	global elapsed_time
	if elapsed_time > 0:
		elapsed_time -= 1

# Indexes for facial landmarks for left and right eye respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
leftEAR = 0
rightEAR = 0
if fileStream:
	# Try video stream from file
	vc = FileVideoStream(args["video"]).start()
	time.sleep(1.0)
else:
	vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	# Forcing resolution increase leads to decrease in frame rate and quality loss
	vc.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH_LANDSCAPE)
	vc.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT_LANDSCAPE)
	time.sleep(1.0)

cv2.namedWindow('Blinker', cv2.WINDOW_FREERATIO)
cv2.setWindowProperty('Blinker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# Video processing main loop
while True:
	if fileStream and not vc.more():
		break
	if fileStream:
		ret, frame = vc.read()
	else:
		ret, frame = vc.read()

	# frame = cv2.flip(frame, 1) # Flip horizontally

	# Get frame dimensions
	orig = frame.copy()
	(origH, origW) = frame.shape[:2]

	# Draw the TP logo at bottom right corner of frame
	tp_logo = cv2.imread('img/tplogo2.png', cv2.IMREAD_UNCHANGED)
	tp_logo_height, tp_logo_width, channels = tp_logo.shape
	width = origW
	height = int(tp_logo_height * (width / tp_logo_width))
	tp_logo = cv2.resize(tp_logo, (width, height))
	orig = overlay_transparent(orig, tp_logo, origW - width, origH - height)

	# Set new width and height then determine ratio in change (faster processing)
	# newW = 1920 # resized frame height for processing
	newW = 1280
	# newW = 640
	newH = int((newW / origW) * origH)
	rW = origW / float(newW)
	rH = origH / float(newH)

	# Resize image and get new dimensions
	frame = cv2.resize(frame, (newW, newH))
	(H, W) = frame.shape[:2]

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
		bothEAR = (leftEAR + rightEAR) / 2

		# Check if minimum number of consecutive frames has passed since EAR reached threshold
		if bothEAR < EYE_AR_THRESHOLD:
			print('Left EAR: {:.4f}, Right EAR: {:.4f}'.format(leftEAR, rightEAR))
			if currently_closed_eye is not 'both':
				currently_closed_eye = 'both'
				frame_counter = 1
			else:
				frame_counter += 1
		elif leftEAR < EYE_AR_THRESHOLD:
			if currently_closed_eye is not 'left':
				currently_closed_eye = 'left'
				frame_counter = 1
			else:
				frame_counter += 1
		elif rightEAR < EYE_AR_THRESHOLD:
			if currently_closed_eye is not 'right':
				currently_closed_eye = 'right'
				frame_counter = 1
			else:
				frame_counter += 1
		else:
			# A blink is detected here
			if frame_counter >= EYE_AR_CONSEC_FRAMES:
				total_blinks += 1

				if currently_closed_eye is 'right':
					# Switch to next category
					nextCategory()
				elif currently_closed_eye is 'left':
					# Set flag to save current frame to disk after specified delay
					executing_screenshot = True
					elapsed_time = screenshot_delay

					for i in range(1, screenshot_delay + 2):
						Timer(i, countdown).start()

				elif currently_closed_eye is 'both':
					if drawableIndex + 1 >= category_item_count:
						drawableIndex = 0
					else:
						drawableIndex += 1


			# Reset counter
			frame_counter = 0
			currently_closed_eye = None


		# Write drawable onto frame
		# Convert rect to a tuple of (startX, startY, endX, endY) as required by drawables.py 'get' function
		startX = int(rect.left() * rW)
		startY = int(rect.top() * rH)
		endX = int(rect.right() * rW)
		endY = int(rect.bottom() * rH)
		bounding_box = (startX, startY, endX, endY)
		if drawDetectionInfo and not executing_screenshot:
			cv2.putText(orig, "Left eye: {:.2f}".format(leftEAR), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)
			cv2.putText(orig, "Right eye: {:.2f}".format(rightEAR), (250, 30), cv2.FONT_HERSHEY_SIMPLEX,
					0.7, (0, 0, 255), 2)
			cv2.putText(orig, "Closed eyes: {}".format(currently_closed_eye), (490, 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)
		
		# DEBUGGING
		bounding_box_width = endX - startX
		# END DEBUGGING
		
		(overlay, x, y) = getByCategory(current_category, drawableIndex, bounding_box, shape)
		width, height = overlay.shape[:2]

		# Calculate angle of head tilt using eye coordinates.
		opposite = abs(leftEye[1][1] - rightEye[1][1])
		adjacent = abs(leftEye[1][0] - rightEye[1][0])
		angle = math.atan(opposite/adjacent) # radians
		angle = math.degrees(angle) # degrees

		if leftEye[1][1] < rightEye[1][1]:
			angle = -angle

		overlay = imutils.rotate_bound(overlay, angle)

		# EXERIMENT BEGIN
		(mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
		mouth = shape[mouthStart:mouthEnd]

		mouth_left_corner = mouth[0]
		mouth_right_corner = mouth[4]
		# for (i, (mX, mY)) in enumerate(mouth):
		# 	color = (255, 0, 0)
		# 	if i == 0 or i == 4:
		# 		color = (0, 255, 0)
		# 	cv2.circle(orig, (mX, mY), 3, color, -1)
		mouth_bottom_center = mouth[6]
		mouth_top_center = mouth[2]
		# cv2.circle(orig, (mouth_left_corner[0], mouth_left_corner[1]), 3, (255, 0, 0), -1)
		# cv2.circle(orig, (mouth_right_corner[0], mouth_left_corner[1]), 3, (0, 255, 0), -1)
		# cv2.circle(orig, (mouth_bottom_center[0], mouth_bottom_center[1]), 3, (0, 0, 255), -1)

		# Calculate length of three sides to get angle using law of cosines
		top = abs(mouth_left_corner[0] - mouth_right_corner[0])
		bottom_left_1 = dist.euclidean(mouth_left_corner, mouth_top_center)
		bottom_right_1 = dist.euclidean(mouth_right_corner, mouth_top_center)
		bottom_left_2 = dist.euclidean(mouth_left_corner, mouth_bottom_center)
		bottom_right_2 = dist.euclidean(mouth_right_corner, mouth_bottom_center)

		angle1 = law_of_cosines_three_known_sides(top, bottom_left_1, bottom_right_1)
		angle2 = law_of_cosines_three_known_sides(top, bottom_left_2, bottom_right_2)
		# EXPERIMENT END

		# A smile is detected when either angle1 is > 7 (subtle smile) or angle2 > 25 (wide smile)

		if drawDetectionInfo and not executing_screenshot and True:
			cv2.putText(orig, "Smile angle (top): {}".format(angle1), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 255, 0), 2)
			cv2.putText(orig, "Smile angle (bottom): {}".format(angle2), (x, y + 50), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 255, 0), 2)
			# cv2.putText(orig, "Angle: {}".format(angle), (x, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 
			# 		0.7, (0, 255, 0), 2)
		if x > 0 and y > 0:
			orig = overlay_transparent(orig, overlay, x, y)

	# Draw categories and highlight current category on the top-right corner of screen
	list_top_y = 10
	list_item_padding = 10
	for category in categories:
		if executing_screenshot:
			break
		color = (0, 255, 0) # default color
		# highlight color
		if category is current_category:
			color = (255, 50, 0)
		text = category
		font = cv2.FONT_HERSHEY_COMPLEX
		font_scale = 0.7
		font_thickness = 2
		(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
		cv2.putText(orig, category, (origW - text_width, list_top_y + text_height + baseline), font, font_scale, color, font_thickness)
		list_top_y += (text_height + baseline + list_item_padding)

	# Rotate final frame to portrait
	if portrait_mode:
		orig = cv2.rotate(orig, cv2.ROTATE_90_CLOCKWISE)

	if executing_screenshot:
		# Show timer on the frame
		if elapsed_time is not 0:
			color = (0, 128, 255) # orange
			text = str(elapsed_time)
			font = cv2.FONT_HERSHEY_COMPLEX
			font_scale = 2
			font_thickness = 5
			(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
			cv2.putText(orig, text, (int(origW / 2 - text_width / 2), int(origH / 2 - text_height / 2)), font, font_scale, color, font_thickness)
		else:
			file_name = str(time.time()) + '.png'
			cv2.imwrite('{}/{}'.format(screenshots_folder_name, file_name), orig)
			executing_screenshot = False
			
			# Indicate screenshot taken
			playsound('sounds/camera-shutter-click.mp3')
			cv2.imshow("Blinker", cv2.rectangle(orig, (0, 0), (origW, origH), (255, 255, 255), -1))
		

	# Show the output frame
	cv2.imshow("Blinker", orig)

	# If specified key is pressed while focus is on the window, end the loop
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	elif key == ord("p"):
		portrait_mode = not portrait_mode
		if portrait_mode:
			w = SCREEN_WIDTH_PORTRAIT
			h = SCREEN_HEIGHT_PORTRAIT
		else:
			w = SCREEN_WIDTH_LANDSCAPE
			h = SCREEN_HEIGHT_LANDSCAPE

		vc.set(cv2.CAP_PROP_FRAME_WIDTH, w)
		vc.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
	elif key == ord("i"):
		drawDetectionInfo = not drawDetectionInfo

# do a bit of cleanup
vc.release()
cv2.destroyAllWindows()
# vs.stop()
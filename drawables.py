import cv2
from scipy.spatial import distance as dist
from imutils import face_utils
from my_utils import law_of_cosines_three_known_sides
from my_utils import eye_aspect_ratio
from my_utils import default_EAR_threshold

# face_utils.FACIAL_LANDMARKS_IDXS contains:
# mouth, inner_mouth, right_eyebrow, left_eyebrow, right_eye, left_eye, nose, jaw

# The only function that should be called from external files!
# Name is defined in the drawables variable
# Regions should be a tuple with (startX, startY, endX, endY) of the face
def get(name, regions):
    (startX, startY, endX, endY) = regions
    drawable = drawables[name]

    # Read the image using imread
    image = cv2.imread(drawable["img"], cv2.IMREAD_UNCHANGED)
    (originalHeight, originalWidth, originalChannels) = image.shape

    # Get the x y coordinates and image dimensions to be drawn on the frame
    (x, y, w, h) = drawable["func"](startX, startY, endX, endY, originalWidth, originalHeight)
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    # Resize image to specified w and h
    dimensions = (w, h)
    image = cv2.resize(image, dimensions)

    return (image, x, y)

def getByCategory(category_name, index, regions, facial_landmarks):
    (startX, startY, endX, endY) = regions
    drawable = categorized_drawables[category_name][index]

    # Check if drawable["img"] is string or function.
    # String should be a path to an image
    # Function should return a string, containing a path to an image
    if callable(drawable["img"]):
        img = drawable["img"](facial_landmarks)
    else:
        img = drawable["img"]

    # Read the image using imread
    image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    (originalHeight, originalWidth, originalChannels) = image.shape

    # Get the x y coordinates and image dimensions to be drawn on the frame
    (x, y, w, h) = drawable["func"](startX, startY, endX, endY, originalWidth, originalHeight, facial_landmarks)
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    # Resize image to specified w and h
    dimensions = (w, h)
    image = cv2.resize(image, dimensions)

    return (image, x, y)


def getDrawableNames():
    return list(drawables.keys())


def getCategory(category_name):
    return categorized_drawables[category_name]


def getCategoryItemCount(category_name):
    print('Category changed. Category: {}, Length: {}'.format(category_name, len(categorized_drawables[category_name])))
    return len(categorized_drawables[category_name])

def getCategoryNames():
    return list(categorized_drawables.keys())

# Takes in an already rescaled dimension, uses the original dimensions to calculate the remaining dimension 
def rescale(width_or_height, rescaledSideOriginal, remainingSideOriginal):
    ratio = width_or_height / rescaledSideOriginal
    remainingSide = remainingSideOriginal * ratio
    return remainingSide

def getCenterX(drawableWidth, faceStartX, faceEndX):
    faceWidth = faceEndX - faceStartX
    faceCenter = faceStartX + (faceWidth / 2)
    x = faceCenter - (drawableWidth / 2)
    return x

def autoScale(dimension, faceWidth):
    divide_ratio = 1.5
    if len(scaling_breakpoints) == 0:
        return dimension
    elif len(scaling_breakpoints) == 1:
        if faceWidth > scaling_breakpoints[0]:
            return dimension
        else:
            dimension = dimension / divide_ratio
            return dimension
    else:
        if faceWidth > scaling_breakpoints[0]:
            return dimension

        for i in range(0, len(scaling_breakpoints)):
            if faceWidth < scaling_breakpoints[i]:
                dimension = dimension / divide_ratio
            else:
                break

        return dimension


################## DEFINE FUNCTIONS TO RETURN COORDINATES FOR EACH DRAWABLE ##################
# Return (x, y, w, h) tuple, where x and y are top-left coordinates, w and h are dimensions of the image to be drawn

def top_of_head(width=None, height=None, offset_from_face=0):
    params = {"width": width, "height": height, "offset_from_face": offset_from_face}
    def func(startX, startY, endX, endY, originalWidth, originalHeight, facialLandmarks):
        width, height, offset_from_face = (params["width"], params["height"], params["offset_from_face"])
        faceWidth = endX - startX
        if width is None and height is None:
            height = 200
            height = autoScale(height, faceWidth)
            width = rescale(height, originalHeight, originalWidth)
        elif height is None:
            width = faceWidth
            height = rescale(width, originalWidth, originalHeight)
        elif width is None:
            height = autoScale(height, faceWidth)
            width = rescale(height, originalHeight, originalWidth)
        else:
            pass

        x = getCenterX(width, startX, endX)
        y = startY - height - offset_from_face
        return (x, y, width, height)
    return func


# Draw on bottom half of face
def bottom_half_of_face(startX, startY, endX, endY, originalWidth, originalHeight, facialLandmarks):
    height = (endY - startY) / 2
    width = endX - startX
    x = startX
    y = endY - height
    return (x, y, width, height)


# Draw the image for the full width of the face, rescaling the height accordingly
def full_face_width(startX, startY, endX, endY, originalWidth, originalHeight, facialLandmarks):
    width = endX - startX
    height = rescale(width, originalWidth, originalHeight)
    x = startX
    faceHeight = endY - startY
    y = startY = (faceHeight - height / 2)
    return (x, y, width, height)

# Draw the image for the full height of the face, rescaling the width accordingly
def full_face_height(startX, startY, endX, endY, originalWidth, originalHeight, facialLandmarks):
    height = endY - startY
    width = rescale(height, originalHeight, originalWidth)
    y = startY
    faceWidth = endX - startX
    x = startX + (faceWidth - width) / 2
    return (x, y, width, height)

def eyes(startX, startY, endX, endY, originalWidth, originalHeight, facialLandmarks):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    leftEye = facialLandmarks[lStart:lEnd]
    width = endX - startX
    height = rescale(width, originalWidth, originalHeight)
    x = startX
    y = leftEye[0][1] - height / 2
    return (x, y, width, height)


def get_image_path_by_emotion(facial_landmarks):
    # Check smiling, frowning or neutral
    (mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
    mouth = facial_landmarks[mouthStart:mouthEnd]
    mouth_left_corner = mouth[0]
    mouth_right_corner = mouth[4]
    mouth_bottom_center = mouth[6]
    mouth_top_center = mouth[2]
    mouth_corners_average_y = (mouth_left_corner[1] + mouth_right_corner[1]) / 2 

    # Calculate length of three sides to get angle using law of cosines.
    # Calculate sides relative to index 2 and index 6 respectively.
    top = abs(mouth_left_corner[0] - mouth_right_corner[0])
    bottom_left_1 = dist.euclidean(mouth_left_corner, mouth_top_center)
    bottom_right_1 = dist.euclidean(mouth_right_corner, mouth_top_center)

    bottom_left_2 = dist.euclidean(mouth_left_corner, mouth_bottom_center)
    bottom_right_2 = dist.euclidean(mouth_right_corner, mouth_bottom_center)
	
    angle1 = law_of_cosines_three_known_sides(top, bottom_left_1, bottom_right_1)
    angle2 = law_of_cosines_three_known_sides(top, bottom_left_2, bottom_right_2)

    # Check if eyes are closed
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = facial_landmarks[lStart:lEnd]
    rightEye = facial_landmarks[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    left_eye_closed = leftEAR < default_EAR_threshold
    right_eye_closed = rightEAR < default_EAR_threshold
    both_eyes_closed = left_eye_closed and right_eye_closed
    no_eyes_closed = not left_eye_closed and not right_eye_closed

    wide_smile = angle2 > 24
    subtle_smile = angle1 > 7 and angle2 > 7 and mouth_corners_average_y < mouth_top_center[1]

	# A smile is detected when either angle1 is > 7 (subtle smile) or angle2 > 25 (wide smile)
    if wide_smile and both_eyes_closed:
        return "img/emojis/both_eyes_closed_wide_smile.png"
    elif wide_smile:
        return "img/emojis/eyes_open_wide_smile.png"
    elif subtle_smile and left_eye_closed:
        return "img/emojis/left_eye_wink_subtle_smile.png"
    elif subtle_smile and no_eyes_closed:
        return "img/emojis/eyes_open_subtle_smile.png"
    else:
        return "img/emojis/neutral_face.png"
        


######################### DATA AND CONFIG #########################
drawables = {
    "miku": {
        "img": "img/miku.png",
        "func": top_of_head()
    },
    "hat": {
        "img": "img/hat.png",
        "func": top_of_head()
    },
    "mask": {
        "img": "img/n95_mask.png",
        "func": bottom_half_of_face
    }
}

categorized_drawables = {
    "Emotion": [
        {
            "img": get_image_path_by_emotion,
            "func": top_of_head()
        }
    ],
    "Animal Ears": [
        {
            "img": "img/animal_ears/animal_ears1.png",
            "func": top_of_head(height=100)
        },
        {
            "img": "img/animal_ears/animal_ears2.png",
            "func": top_of_head(height=100)
        },
        {
            "img": "img/animal_ears/animal_ears3.png",
            "func": top_of_head(height=200)
        },
        {
            "img": "img/animal_ears/animal_ears4.png",
            "func": top_of_head(height=150)
        },
        {
            "img": "img/animal_ears/animal_ears5.png",
            "func": top_of_head(height=150)
        },
        {
            "img": "img/animal_ears/animal_ears6.png",
            "func": top_of_head(height=150)
        },
        {
            "img": "img/animal_ears/animal_ears7.png",
            "func": top_of_head(height=150)
        },
        {
            "img": "img/animal_ears/animal_ears8.png",
            "func": top_of_head(height=150)
        }
    ],
    "Eyewear": [
        {
            "img": "img/eyewear/eyewear1.png",
            "func": eyes
        },
        {
            "img": "img/eyewear/eyewear2.png",
            "func": eyes
        },
        {
            "img": "img/eyewear/eyewear3.png",
            "func": eyes
        },
        {
            "img": "img/eyewear/eyewear4.png",
            "func": eyes
        },
        {
            "img": "img/eyewear/eyewear5.png",
            "func": eyes
        },
        {
            "img": "img/eyewear/eyewear6.png",
            "func": eyes
        },
        {
            "img": "img/eyewear/eyewear7.png",
            "func": eyes
        },
        {
            "img": "img/eyewear/eyewear8.png",
            "func": eyes
        },
        {
            "img": "img/eyewear/eyewear9.png",
            "func": eyes
        },
        {
            "img": "img/eyewear/eyewear10.png",
            "func": eyes
        },
    ],
    "Hats": [
        {
            "img": "img/hats/hat1.png",
            "func": top_of_head(offset_from_face=-10)
        },
        {
            "img": "img/hats/hat2.png",
            "func": top_of_head(height=200, offset_from_face=200/4)
        },
        {
            "img": "img/hats/hat3.png",
            "func": top_of_head()
        },
        {
            "img": "img/hats/hat4.png",
            "func": top_of_head()
        },
        {
            "img": "img/hats/hat5.png",
            "func": top_of_head()
        },
        {
            "img": "img/hats/hat6.png",
            "func": top_of_head()
        },
        {
            "img": "img/hats/hat7.png",
            "func": top_of_head()
        },
        {
            "img": "img/hats/hat8.png",
            "func": top_of_head()
        },
    ],
    "Anime": [
        {
            "img": "img/anime/anime1.png",
            "func": top_of_head()
        },
        {
            "img": "img/anime/anime2.png",
            "func": top_of_head()
        },
        {
            "img": "img/anime/anime3.png",
            "func": top_of_head()
        },
        {
            "img": "img/anime/anime4.png",
            "func": top_of_head()
        },
        {
            "img": "img/anime/anime5.png",
            "func": top_of_head()
        },
        {
            "img": "img/anime/anime6.png",
            "func": top_of_head()
        },
        {
            "img": "img/anime/anime7.png",
            "func": top_of_head()
        },
        {
            "img": "img/anime/anime8.png",
            "func": top_of_head()
        },
        {
            "img": "img/anime/anime9.png",
            "func": top_of_head()
        },
        {
            "img": "img/anime/anime10.png",
            "func": top_of_head()
        },
    ]
}

# Scaling breakpoints in DESCENDING ORDER
scaling_breakpoints = [150, 120, 90, 60]
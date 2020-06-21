import cv2
from imutils import face_utils

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

    # Read the image using imread
    image = cv2.imread(drawable["img"], cv2.IMREAD_UNCHANGED)
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

def top_of_head(width=None, height=None):
    dimensions = {"width": width, "height": height}
    def func(startX, startY, endX, endY, originalWidth, originalHeight, facialLandmarks):
        width, height = (dimensions["width"], dimensions["height"])
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
        y = startY - height - (height / 4)
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
    y = leftEye[1][1]
    return (x, y, width, height)


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
    "Glasses": [
        {
            "img": "img/thug_life_glasses.png",
            "func": eyes
        }
    ],
    "Animal Ears": [
        {
            "img": "img/dog_ears.png",
            "func": top_of_head(height=150)
        },
        {
            "img": "img/car_ears.png",
            "func": top_of_head(height=150)
        },
        {
            "img": "img/bunny_ears_right_down.png",
            "func": top_of_head()
        }
    ],
    "Party": [
        {
            "img": "img/miku.png",
            "func": top_of_head()
        },
        {
            "img": "img/hat.png",
            "func": top_of_head()
        },
        {
            "img": "img/n95_mask.png",
            "func": bottom_half_of_face
        }
    ]
}

# Scaling breakpoints in DESCENDING ORDER
scaling_breakpoints = [150, 120, 90, 60]
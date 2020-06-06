import cv2

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

def getDrawableNames():
    return list(drawables.keys())

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

# Draw miku on top of head!
def miku(startX, startY, endX, endY, originalWidth, originalHeight):
    # MAX height
    height = 200
    faceWidth = endX - startX
    height = autoScale(height, faceWidth)
    # Break off points to scale down the image size
    width = rescale(height, originalHeight, originalWidth)
    x = getCenterX(width, startX, endX)
    y = startY - height - (height / 4)
    return (x, y, width, height)

# Draw hat on top of head
def hat(startX, startY, endX, endY, originalWidth, originalHeight):
    # MAX height
    height = 200
    faceWidth = endX - startX
    height = autoScale(height, faceWidth)
    width = rescale(height, originalHeight, originalWidth)
    x = getCenterX(width, startX, endX)
    y = startY - height - (height / 4)
    return (x, y, width, height)

# Draw surgical mask on bottom half of face 
def mask(startX, startY, endX, endY, originalWidth, originalHeight):
    height = (endY - startY) / 2
    width = endX - startX
    x = startX
    y = endY - height
    return (x, y, width, height)


drawables = {
    "miku": {
        "img": "img/miku.png",
        "func": miku
    },
    "hat": {
        "img": "img/hat.png",
        "func": hat
    },
    "mask": {
        "img": "img/n95_mask.png",
        "func": mask
    }
}

# Scaling breakpoints in DESCENDING ORDER
scaling_breakpoints = [150, 120, 90, 60]
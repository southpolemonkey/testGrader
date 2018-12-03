import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-b", "--background", required=True, help="background image")
ap.add_argument('--method', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = vars(ap.parse_args())

bg = cv2.imread(args['background'])
image = cv2.imread(args['image'])
bg = cv2.resize(bg, (600, 1000))
image = cv2.resize(image, (600, 1000))

# Approach one: subtraction between two numpy arrays
diff = abs(image-bg)
cv2.imwrite("diff.png", diff)


# Approach two: background subtraction
# https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
if args['method'] == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()

fgMask = backSub.apply(bg)

cv2.imshow('bg', bg)
cv2.imshow('FG Mask', fgMask)
cv2.waitKey(0)





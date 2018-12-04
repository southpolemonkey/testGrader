import numpy as np
import cv2 as cv
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

img = cv.imread(args['image'])
# cv.imshow('origin', img)
# cv.waitKey(0)


crop_img = img[383:1590, 177:1029]
student_number = img[186:235, 560:1040]
cv.imshow('cropped', crop_img)
cv.imshow('student number', student_number)

cv.waitKey(0)

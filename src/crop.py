import numpy as np
import cv2 as cv
import argparse
from markBubble import get_answer

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

img = cv.imread(args['image'])
# cv.imshow('origin', img)
# cv.waitKey(0)
crop_img = img[383:1590, 177:1029]
student_number = img[186:235, 560:1040]


def preprocessed(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    return thresh


# TODO: 1) recoginze student ID, pass SID as paramter to marker function
get_answer(crop_img, preprocessed(crop_img))
# cv.imshow('cropped', crop_img)
# cv.imshow('student number', student_number)
# cv.waitKey(0)


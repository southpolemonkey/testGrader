import numpy as np
import cv2 as cv
import argparse
from markBubble import get_answer
from sid import read_student_id
import requests
import json
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

GROUND_TRUTH = [5, 6, 7, 8, 9, 8, 7, 7, 2]

ori = cv.imread(args['image'])
img = cv.resize(ori, (1239, 1752))

crop_img = img[383:1590, 303:1029]
student_number = img[165:235, 560:1050]


def preprocessed(image, flag=False):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    if flag:
        cnts = cv.findContours(thresh.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        corner = None
        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:4]

            # loop over the sorted contours
            for c in cnts:
                # approximate the contour
                peri = cv.arcLength(c, True)
                approx = cv.approxPolyDP(c, 0.02 * peri, True)

                if len(approx) == 4:
                    corner = np.asarray(approx)
                    break
        return corner
    else:
        return thresh


def model_accuracy(test_file):
    matches = np.asarray(test_file) == GROUND_TRUTH
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/len(test_file)
    return accuracy


# cv.imshow('cropped', crop_img)
# cv.imshow('sid', student_number)
# cv.waitKey(0)

get_answer(crop_img, preprocessed(crop_img))
sid = read_student_id(student_number)
accuracy = model_accuracy(sid)

print(f"Ground Truth: {GROUND_TRUTH}")
print(f"Prediction: {sid}")
print(f"Accuracy: {accuracy:.2f}%")


# TODO: Deploy keras model in flask server
# get prediction from flask sever
# r = requests.post('http://localhost:6500/predict', files={'files': json.dumps(sid.tolist())})
# print(r.content)




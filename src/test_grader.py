# USAGE
# python test_grader.py --image images/test_01.png

# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# define the answer key which maps the question number
# to the correct answer
# ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
ANSWER_KEY = {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 1, 6: 2, 7: 2, 8: 1, 9: 3, 10: 0,
              11: 3, 12: 1, 13: 2, 14: 1, 15: 2, 16: 2, 17: 0, 18: 2, 19: 1}
ANSWER = {0: 'A', 1: 'B', 2: 'C', 3: "D"}

# load the image, convert it to grayscale, blur it
# slightly, then find edges
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# edged = cv2.Canny(blurred, 75, 200)

# find contours in the edge map, then initialize
# the contour that corresponds to the document
# cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
# docCnt = None

# TODO: make the algorithm find the whole paper edge, instead of student number box
# TODO: feed student number box to Keras to do digit recognition

# ensure that at least one contour was found
# if len(cnts) > 0:
#     # sort the contours according to their size in
#     # descending order
#     cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
#
#     # loop over the sorted contours
#     for c in cnts:
#         # approximate the contour
#         peri = cv2.arcLength(c, True)
#         approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#
#         # if our approximated contour has four points,
#         # then we can assume we have found the paper
#         if len(approx) == 4:
#             docCnt = approx
#             break


# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper
# paper = four_point_transform(image, docCnt.reshape(4, 2))
# warped = four_point_transform(gray, docCnt.reshape(4, 2))

# TODO: next step is to extract box by area, ideally the first two should be paper edge and student number box
# id_paper = four_point_transform(image, docCnt[2].reshape(4, 2))
# id_warped = four_point_transform(gray, docCnt[2].reshape(4, 2))

# cv2.imshow('Tranform-answer', ans_warped)
# cv2.imshow('Tranform-id', id_warped)
# cv2.waitKey(0)

# apply Otsu's thresholding method to binarize the warped piece of paper
# thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imshow('thresh.png', thresh)
cv2.waitKey(0)


# find contours in the threshold image, then initialize
# the list of contours that correspond to questions
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[0] if imutils.is_cv2() else cnts[1]
questionCnts = []

# loop over the contours
for c in cnts:
    # compute the bounding box of the contour, then use the
    # bounding box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # in order to label the contour as a question, region
    # should be sufficiently wide, sufficiently tall, and
    # have an aspect ratio approximately equal to 1.5
    if w >= 25 and h >= 15 and ar >= 1.2:
        questionCnts.append(c)
        # cv2.drawContours(image, c, -1, (0, 255, 0), 2)
# cv2.imshow('bubble', image)
# cv2.waitKey(0)

# sort the question contours top-to-bottom, then initialize
# the total number of correct answers
questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
correct = 0

# each question has 4 possible answers, to loop over the
# question in batches of 4
for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
    # sort the contours for the current question from
    # left to right, then initialize the index of the
    # bubbled answer
    cnts = contours.sort_contours(questionCnts[i:i + 4])[0]

    bubbled = None

    # loop over each row
    for (j, c) in enumerate(cnts):
        # construct a mask that reveals only the current
        # "bubble" for the question
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # apply the mask to the threshold image, then
        # count the number of non-zero pixels in the
        # bubble area
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        # if the current total has a larger number of total
        # non-zero pixels, then we are examining the currently
        # bubbled-in answer
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

    # initialize the contour color and the index of the
    # *correct* answer
    color = (0, 0, 255)
    k = ANSWER_KEY[q]

    # check to see if the bubbled answer is correct
    if k == bubbled[1]:
        print(f'No.{q+1} correct')
        color = (0, 255, 0)
        correct += 1
    # print(f'Question{q+1}: {ANSWER[bubbled[1]]}')

    # draw the outline of the correct answer on the test
    cv2.drawContours(image, [cnts[k]], -1, color, 3)

# grab the test taker
score = (correct / len(ANSWER_KEY)) * 100
with open('output/record.txt', 'w') as f:
    print("[INFO] score: {:.2f}%".format(score))
    f.write(str(score))

cv2.putText(image, "{:.2f}%".format(score), (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
# cv2.imshow("Original", image)
cv2.imwrite("output/marked.png", image)
# cv2.waitKey(0)
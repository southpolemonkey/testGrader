from imutils import contours
import numpy as np
import imutils
import cv2
import os

ANSWER_KEY = {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 1, 6: 2, 7: 2, 8: 1, 9: 3, 10: 0,
              11: 3, 12: 1, 13: 2, 14: 1, 15: 2, 16: 2, 17: 0, 18: 2, 19: 1}
ANSWER = {0: 'A', 1: 'B', 2: 'C', 3: "D"}


def get_answer(image, thresh):
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

    # sort the question contours top-to-bottom, then initialize
    # the total number of correct answers
    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    correct = 0

    if not os.path.exists('../output/'):
        os.mkdir('../output/')
    f = open('../output/record.txt', 'w')

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
            color = (0, 255, 0)
            correct += 1
            f.write(f'No.{q+1} correct \n')
        else:
            f.write(f'No.{q+1} incorrect \n')

        # draw the outline of the correct answer on the test
        cv2.drawContours(image, [cnts[k]], -1, color, 3)

    # grab the test taker
    score = (correct / len(ANSWER_KEY)) * 100
    f.write(str(score) + '%')
    f.close()

    # save the marked sheet
    cv2.putText(image, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite("../output/marked.png", image)


if __name__ == 'main':
    pass
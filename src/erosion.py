import cv2
import numpy as np
import argparse
import math


# create a black mask
def mask():
    mask = img.copy()
    for i in range(3):
        mask[520:3220, 250:2200, i] = 0


def rotate_image(thresh, img):
    ymin = [None, None]
    ymax = [None, None]
    xmin = [None, None]
    xmax = [None, None]

    # find min max of x coordinate
    for x in range(thresh.shape[1]):
        for y in range(thresh.shape[0]):
            if thresh[y, x] != 255:
                continue
            if xmin[0] is None or x < xmin[0]:
                xmin[0] = x
                xmin[1] = y
            elif xmax[0] is None or x > xmax[0]:
                xmax[0] = x
                xmax[1] = y

            if ymin[1] is None or y < ymin[1]:
                ymin[0] = x
                ymin[1] = y
            elif ymax[1] is None or y > ymax[1]:
                ymax[0] = x
                ymax[1] = y
    # cv2.circle(img, tuple(ymin), radius=5, color=(0, 0, 255))
    # cv2.circle(img, tuple(xmin), radius=5, color=(0, 0, 255))
    # cv2.circle(img, tuple(ymax), radius=5, color=(0, 0, 255))
    # cv2.circle(img, tuple(xmax), radius=5, color=(0, 0, 255))

    if xmin[1] < xmax[1]:
        degree = -math.atan((ymax[0] - xmin[0])/(ymax[1] - xmin[1]))
    else:
        degree = math.atan((ymin[0]-xmin[0])/abs(xmin[1]-ymin[1]))

    # tl, br = [xmin, ymax] if xmin[1] < xmax[1] else [xmin, ymin]
    # degree = math.atan2(tl[1] - br[1], tl[0] - br[0])
    print(f'Rotation degree: {math.degrees(degree)}')
    M = cv2.getRotationMatrix2D(tuple(ymax), math.degrees(degree), 1)
    rows, cols, channels = img.shape
    cv2.warpAffine(img, M, (cols, rows), img, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return img

# # The Standard Hough Transform
# cdst = img.copy()
# lines = cv2.HoughLines(erosion, 1, np.pi / 180, 150, None, 0, 0)
# if lines is not None:
#     for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         a = math.cos(theta)
#         b = math.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
#         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
#         cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)


def probabilistic_hough_line_transform(erosion, img):
    # The Probabilistic Hough Line Transform
    cdstP = img.copy()
    linesP = cv2.HoughLinesP(erosion, 1, np.pi / 180, 50, None, 50, 10)
    min_x = img.shape[1]
    min_y = img.shape[0]
    max_x = 0
    max_y = 0
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            # update x coordinate
            if min(l[0], l[2]) < min_x:
                min_x = min(l[0], l[2])
                tl = [min_x, min(l[1], l[3])]
            if max(l[0], l[2]) > max_x:
                max_x = max(l[0], l[2])
                br = [max_x, max(l[1], l[3])]
            # update y coordinate
            if min(l[1], l[3]) < min_y:
                min_y = min(l[1], l[3])
            if max(l[1], l[3]) > max_y:
                max_y = max(l[1], l[3])
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
    extract = img[min_y:max_y, min_x:max_x]
    return extract


def test_marker(img):
    return


def keras_recognition(img):
    # TODO: get prediction from keras model
    accuracy = []
    return


def main(image_path):
    img = cv2.imread(image_path).copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    print(f'The shape of thresh: {thresh.shape}')

    rotate_image(thresh, img)

    # eroding threshed image
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=1)

    extract_part = probabilistic_hough_line_transform(rotated_img. img)

    bubbles = extract_part[:, :]
    student_number = extract_part[:, :]

    accuracy = test_marker(bubbles)
    sid = keras_recognition(student_number)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    args = vars(ap.parse_args())
    img_path = args['image']
    main(img_path)

    # cv2.imshow('origin image', img)
    # cv2.imshow('mask', mask)
    # cv2.imshow('thresh', thresh)
    # cv2.imshow('erosion', erosion)
    # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    # cv2.imshow("Extracted part", extract)
    # output_file = args['image'][:-4] + '_edited.png'
    # rotated_img = rotate_image()
    # cv2.imwrite(output_file, rotated_img)

    while True:
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


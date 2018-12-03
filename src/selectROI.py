import cv2
import numpy as np

if __name__ == '__main__':
    # Read image
    im = cv2.imread("images/12.png")

    im = cv2.resize(im,(0,0), fx=0.5, fy=0.5)
    fromCenter = False
    r = cv2.selectROIs('Select ROIs', im, fromCenter)

    try:
        imCrop_1 = im[int(r[0][1]):int(r[0][1] + r[0][3]), int(r[0][0]):int(r[0][0] + r[0][2])]
        imCrop_2 = im[int(r[1][1]):int(r[1][1] + r[1][3]), int(r[1][0]):int(r[1][0] + r[1][2])]
        print(imCrop_1.shape)
        print(imCrop_2.shape)
        cv2.imwrite("Image1.png", imCrop_1)
        cv2.imwrite("Image2.png", imCrop_2)
        cv2.waitKey(0)
    except IndexError:
        print("Not enough areas selected.")


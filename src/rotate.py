import cv2

img = cv2.imread('../images/sheet.png')

rows, cols, channels = img.shape

M = cv2.getRotationMatrix2D((cols/2,rows/2),-15,0.75)
dst = cv2.warpAffine(img, M, (cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

cv2.imwrite('rotate_2.png', dst)

# import numpy as np
# import cv2 as cv
# import json
#
#
#
# def toGray(img):
#     return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# def threSholding(grayImage):
#     return cv.threshold(grayImage, 127, 255, cv.THRESH_BINARY)
#
# def adaptiveThresholding(grayImage):
#     return cv.adaptiveThreshold(grayImage, 255, 1,\
#                                     1, 11, 2)
#
# def gussianBlur(grayImage):
#     return cv.GaussianBlur(grayImage, (5,5), 0)
#
# def findContour(imageProcess):
#     return cv.findContours(imageProcess, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
# def erosionImage(threshodingImage):
#     kernel = np.ones((5,5), np.uint8)
#     return cv.erode(threshodingImage, kernel, iterations=1)
#
# def dilationImage(thresholdingImage):
#     kernel = np.ones((3,3), np.uint8)
#     return cv.dilate(thresholdingImage, kernel, iterations=1)
#
# def openingImage(thresholdingImage):
#     kernel = np.ones((2,2), np.uint8)
#     return cv.morphologyEx(thresholdingImage, cv.MORPH_OPEN, kernel)
#
# def processImage(copyImage):
#
#     grayImage = toGray(copyImage)
#     blur = gussianBlur(grayImage)
#     cv.imshow("Blur", blur)
#     mask = np.zeros((grayImage.shape), np.uint8)
#     imageProcess = adaptiveThresholding(blur)
#     cv.imshow("AdaptiveThresholdingImage", imageProcess)
#
#     contours, hierarchy, = findContour(imageProcess)
#     cnt = sorted(contours, key=cv.contourArea, reverse=True)
#
#     approx = np.array([])
#     sodukuContour = np.array([])
#     for idx,cont in enumerate(cnt):
#         # cv.contourArea(cnt[idx + 1]) < cv.contourArea(cont) and
#         if cv.contourArea(cont) > 15000:
#             perimeter = cv.arcLength(cont, True)
#             approx = cv.approxPolyDP(cont, 0.02 * perimeter, True)
#             if len(approx) == 4:
#                 sodukuContour = cont
#                 break
#     return sodukuContour, approx, mask, blur
#
# def perspictiveImage(approx):
#     img = frame.copy()
#     approx = np.squeeze(approx)
#     fourPoit = np.zeros((4, 2))
#     sumAprrox = np.sum(approx, axis=1)
#     diffApprox = np.diff(approx, axis=1)
#
#     fourPoit[0] = approx[np.argmin(sumAprrox)]
#     fourPoit[3] = approx[np.argmax(sumAprrox)]
#     fourPoit[1] = approx[np.argmin(diffApprox)]
#     fourPoit[2] = approx[np.argmax(diffApprox)]
#
#     pts1 = np.float32([fourPoit[0], fourPoit[1], fourPoit[2], fourPoit[3]])
#     pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
#     M = cv.getPerspectiveTransform(pts1, pts2)
#     img = cv.warpPerspective(img, M, (450, 450))
#     return img
#
# def findingVerticallines(img):
#     kernelv = cv.getStructuringElement(cv.MORPH_RECT, (2, 10))
#     dx = cv.Sobel(img, cv.CV_16S, 1, 0)
#     dx = cv.convertScaleAbs(dx)
#     cv.normalize(dx, dx, 0, 255, cv.NORM_MINMAX)
#     ret, close = cv.threshold(dx, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#     dx = cv.morphologyEx(close, cv.MORPH_DILATE, kernelv, iterations=1)
#     contour, hier = cv.findContours(dx, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     for cnt in contour:
#         x, y, w, h = cv.boundingRect(cnt)
#         if h / w > 5:
#             cv.drawContours(dx, [cnt], 0, 255, -1)
#         else:
#             cv.drawContours(dx, [cnt], 0, 0, -1)
#     close = cv.morphologyEx(dx, cv.MORPH_CLOSE, None, iterations=2)
#     closex = close.copy()
#     return closex
#
#
# obj = cv.VideoCapture(0)
#
#
# while True:
#     maxArea = 0
#     indexContor = 0
#     ret, frame = obj.read()
#     copyImage = frame.copy()
#
#
#
#     sodukuContour, approx, mask, blurGray = processImage(copyImage)
#     res = cv.adaptiveThreshold(blurGray, 255, 0, 1, 11, 2)
#     cv.drawContours(mask, sodukuContour, -1, 255, 9)
#     # cv.drawContours(mask, sodukuContour, -1, 0, 9)
#     # cv.drawContours(mask, sodukuContour, 0, 255, -1)
#     # cv.drawContours(mask,sodukuContour,0,0,2)
#
#     cv.imshow("1", mask)
#     # res = cv.bitwise_and(adaptiveimage, mask)
#     img1 = cv.drawContours(frame.copy(), sodukuContour, -1, (0, 0, 255), 9)
#     cv.imshow("With Contour", img1)
#     # mask= cv.bitwise_and(img1, mask)
#     cv.imshow("Mask", res)
#
#     # vertical_image = findingVerticallines(mask)
#     # cv.imshow("vertical", vertical_image)
#
#
#
#     if len(approx) == 4:
#         img = perspictiveImage(approx)
#         cv.imshow("with 4Pointes", img)
#
#
#         grayfourpoint = toGray(img)
#         adaptive4point = adaptiveThresholding(grayfourpoint)
#
#         # cv.imshow("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH", openingImage(adaptive4point))
#
#
#         # delete vertical and horizontal lines
#         for i in range(10):
#             if i == 0 :
#                 adaptive4point[ (i * 50) : (i * 50) + 10 , : ] = 0
#                 adaptive4point[:, (i * 50): (i * 50) + 12] = 0
#             else :
#                 adaptive4point[(i * 50) - 4: (i * 50) + 10, :] = 0
#                 adaptive4point[:, (i * 50) - 4: (i * 50) + 12] = 0
#
#
#         # adaptive4point = cv.erode(adaptive4point, (3,3))
#         # adaptive4point = openingImage(adaptive4point)
#         # adaptive4point = openingImage(adaptive4point)
#         cv.imshow("adaptive4Points", adaptive4point)
#
#
#
#         allCells = np.zeros((81,50,50))
#         idx = 0
#         for i in range(9):
#             for j in range(9):
#                 allCells[idx] = adaptive4point[i*50 : (i*50) + 50 , j*50 : (j*50) + 50]
#                 idx += 1
#         for idx,img in enumerate(allCells):
#             img = openingImage(img)
#             # img = openingImage(img)
#             cv.imwrite('C:/Users/ASUS/Desktop/ComputerVisionProject/Number/{}.png'.format(idx+1), img)
#
#         idx = 0
#
#         # for digit in allCells:
#         #     print(digit.shape)
#         #     sift = cv.SIFT_create()
#         #     kp = sift.detect(digit, None)
#         #     img2 = cv.drawKeypoints(digit, kp, None, color=(0, 255, 0))  # without assign
#         #     cv.imwrite('Number features/{}.png'.format(idx+1), img2)
#         #     idx += 1
#
#
#
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break


import cv2
from imutils import contours
import numpy as np

# Load image, grayscale, and adaptive threshold
image = cv2.imread('3213.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,57,5)

# Filter out all numbers and noise to isolate only boxes
cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 1000:
        cv2.drawContours(thresh, [c], -1, (0,0,0), -1)

# Fix horizontal and vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)

# Sort by top to bottom and each row by left to right
invert = 255 - thresh
cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
(cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

sudoku_rows = []
row = []
for (i, c) in enumerate(cnts, 1):
    area = cv2.contourArea(c)
    if area < 50000:
        row.append(c)
        if i % 9 == 0:
            (cnts, _) = contours.sort_contours(row, method="left-to-right")
            sudoku_rows.append(cnts)
            row = []

# Iterate through each box
for row in sudoku_rows:
    for c in row:
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, (255,255,255), -1)
        result = cv2.bitwise_and(image, mask)
        result[mask==0] = 255
        cv2.imshow('result', result)
        cv2.waitKey(175)

cv2.imshow('thresh', thresh)
cv2.imshow('invert', invert)
cv2.waitKey()
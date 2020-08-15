from point_transform import four_point_transform

import numpy as np
import argparse
import cv2
import imutils


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# Run edge detection setup (grayscale, blur, canny edge detection) for contour detection. Scale to 500 px to make detection easier
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# Find countours with assumption largest contour with four points is document
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5] # gets 5 biggest contours by area in descending order

for cnt in cnts:
    peri = cv2.arcLength(cnt, True) 
    # multiply contour length by 0.02 to get precision (how accurate polynomial approximates contour) 
    # https://theailearner.com/2019/11/22/simple-shape-detection-using-contour-approximation/
    approx = cv2.approxPolyDP(cnt, peri * 0.02, True)

    if len(approx) == 4:
        # found contour representing document
        screenCnt = approx
        break

# Transform four point contour and return
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
cv2.imwrite("output.png", warped)
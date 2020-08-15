import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
copy = image.copy()

for cnt in cnts:
    cv2.drawContours(copy, [cnt], -1, (240, 0, 159), 3)

mask = thresh.copy()
copy = cv2.bitwise_and(copy, thresh, mask=mask)
    


output = copy
cv2.imwrite("output.png", output)


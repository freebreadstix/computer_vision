"""
Given (x, y) coordinates representing corners of document to extract,
return image with document transformed to take entire frame
"""

import numpy as np
import cv2


def __order_points(pts):
    """
    Orders points for transformation clockwise from top-left
    returns: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")

    # sum coordinates, smallest and largest are top left and bottom right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # diff coordinates, smallest and alrgest are top-right, bottom-left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    # sort points
    rect = __order_points(pts)
    (tl, tr, br, bl) = rect

    # get largest height, width differences
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # create destination array representing size of image
    dest_arr =  np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")

    # get transformation matrix and apply perspective transform
    M = cv2.getPerspectiveTransform(rect, dest_arr)
    transformed = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return transformed





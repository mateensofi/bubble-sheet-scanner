# Necessary Imports
import numpy as np
import argparse
import cv2
import imutils
from imutils import contours
from imutils.perspective import four_point_transform


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# Define the answer key which maps the question number
# to the correct answer
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

# Image loading and Edge Detection
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
edged = cv2.Canny(blurred, threshold1=75, threshold2=200)

# Find contours in the edge map
cnts = cv2.findContours(edged.copy(), mode=cv2.RETR_EXTERNAL,
                        method=cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Initialize the contour that corresponds to document
doc_contour = None

# Ensure that at least one contour was found
if len(cnts) > 0:
    # Sort the contours according to their size
    # in descending order
    sorted(cnts, key=cv2.contourArea, reverse=True)

    # Loop over the sorted contours
    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            doc_contour = approx
            break

# Apply the four point transform to both the
# original image and gray image
paper = four_point_transform(image, doc_contour.reshape(4, 2))
warped = four_point_transform(gray, doc_contour.reshape(4, 2))

# Apply Otsu's thresholding method to binarize the image
thresh = cv2.threshold(warped, 0, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Find contours in the thresh image and initialize
# the list of contours that correspond to questions
cnts = cv2.findContours(thresh.copy(), mode=cv2.RETR_EXTERNAL,
                        method=cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
question_contours = list()

# Loop over the contours
for c in cnts:
    # Compute the bounding box of the contour and
    # use the bounding box to derive the aspect ratio
    (x, y, h, w) = cv2.boundingRect(c)
    ar = w / float(h)

    # In order to label the contour as a question, region
    # should be sufficiently wide, sufficiently tall, and
    # have an aspect ratio approximately equal to 1
    if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
        question_contours.append(c)

# Sort the question contours top-to-bottom
question_contours = contours.sort_contours(question_contours, method="top-to-bottom")[0]
correct = 0

# Loop over the contours in batches of five
# as each question has five possible answers
for (q, i) in enumerate(np.arange(0, len(question_contours), 5)):
    # Sort the contours for the current question from left-to-right
    cnts = contours.sort_contours(question_contours[i:i+5], method="left-to-right")[0]
    bubbled = None

    # Determine which bubble is filled in
    for (j, c) in enumerate(cnts):
        # construct a mask that reveals only the
        # current bubble for the question
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # Apply the mask to thresh image, then count
        # the number of non-zero pixels in the bubble area
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        # If the current total has a larger number of total
        # non-zero pixels, then we are examining the currently
        # bubbled-in answer
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

    # Initialize the contour color and the
    # index of the correct answer
    color = (0, 0, 255)
    k = ANSWER_KEY[q]

    # Check to see if the bubbled answer is correct
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1

    # Draw the outline of the correct answer on the test
    cv2.drawContours(paper, [cnts[k]], -1, color, 3)

# Grab the test taker
score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)

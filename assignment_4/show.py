import sys
import cv2

cv2.imshow(sys.argv[1], cv2.imread(sys.argv[1]))
cv2.waitKey()
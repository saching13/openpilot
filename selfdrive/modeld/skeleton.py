import cv2
import numpy as np


# Example usage:
img = cv2.imread('path_to_mask_image.png', 0)
_, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
skeleton = skeletonize(binary_img)

cv2.imshow('Original', binary_img)
cv2.imshow('Skeleton', skeleton)
cv2.waitKey(0)
cv2.destroyAllWindows()

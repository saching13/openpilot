import cv2
import numpy as np

def update(val=None):
    # Get values from trackbars
    h1 = cv2.getTrackbarPos('Hue Lower', 'Masking')
    h2 = cv2.getTrackbarPos('Hue Upper', 'Masking')
    s1 = cv2.getTrackbarPos('Saturation Lower', 'Masking')
    s2 = cv2.getTrackbarPos('Saturation Upper', 'Masking')
    v1 = cv2.getTrackbarPos('Value Lower', 'Masking')
    v2 = cv2.getTrackbarPos('Value Upper', 'Masking')
    
    lower_bound = np.array([h1, s1, v1])
    upper_bound = np.array([h2, s2, v2])

    # Create a mask based on trackbar values
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)
    print(f'Lower bound -> {lower_bound}')
    print(f'Upper bound {upper_bound}')
    # Show the mask and result
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

# Load the image
image = cv2.imread('dataset/capture_7.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create a window
cv2.namedWindow('Masking')

# Create trackbars for adjusting the lower and upper HSV values
cv2.createTrackbar('Hue Lower', 'Masking', 0, 179, update)
cv2.createTrackbar('Hue Upper', 'Masking', 179, 179, update)
cv2.createTrackbar('Saturation Lower', 'Masking', 0, 255, update)
cv2.createTrackbar('Saturation Upper', 'Masking', 255, 255, update)
cv2.createTrackbar('Value Lower', 'Masking', 0, 255, update)
cv2.createTrackbar('Value Upper', 'Masking', 255, 255, update)

# Call the update function once to display the initial images
update()

cv2.waitKey(0)
cv2.destroyAllWindows()

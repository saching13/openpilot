import cv2
import numpy as np
import glob

import cv2
import numpy as np

def find_vanishing_point(img):
    # Load the image and convert to grayscale
    # img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None:
        print("No lines were detected")
        return

    # Draw the lines and find their intersection points
    intersections = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Get intersection points
        for other_line in lines:
            if other_line is not line:
                rho2, theta2 = other_line[0]
                if abs(theta - theta2) > 0.01:  # Avoid parallel lines
                    A = np.array([
                        [a, -np.cos(theta2)],
                        [b, -np.sin(theta2)]
                    ])
                    B = np.array([[rho], [rho2]])
                    intersection = np.linalg.solve(A, B)
                    intersections.append(intersection)

    # Approximate the vanishing point as the mean of intersection points
    if intersections:
        vx, vy = np.mean(intersections, axis=0).squeeze()
        cv2.circle(img, (int(vx), int(vy)), 10, (0, 255, 0), -1)
    return img
    # cv2.imshow('Vanishing Point', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




image_files = glob.glob("dataset" + "/*.jpg")
for image_file in image_files:
    # Load the image
    image = cv2.imread(image_file)

    # Convert the image to HSV format
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the hue, saturation, and value
    # For example, to detect green color
    # lower_bound = np.array([12, 61,  0])  # Lower bound for green hue
    # upper_bound = np.array([166, 255, 129])  # Upper bound for green hue
    lower_bound = np.array([4, 114,  59])  # Lower bound for green hue
    upper_bound = np.array([177, 156, 100])  # Upper bound for green hue
    # Create a mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # (Optional) Bitwise-AND the mask and original image
    result = cv2.bitwise_and(image, image, mask=mask)

    # Show the original image, mask, and result
    vanishing = find_vanishing_point(result)

    cv2.imshow('Vanishing', vanishing)
    cv2.imshow('Original Image', image)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
cv2.destroyAllWindows()


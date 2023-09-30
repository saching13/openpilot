import cv2
import numpy as np
import glob

def detect_x(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # You can add more logic to check if any two lines intersect and form an "X"

    cv2.imshow('Detected X', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_files = glob.glob("dataset" + "/*.jpg")

for image_file in image_files:
    detect_x(image_file)
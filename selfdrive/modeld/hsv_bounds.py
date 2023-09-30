import cv2
import numpy as np
import glob

# intrinsic = np.array([[567.0, 0.0, 1928.0 / 2],
#                       [0.0, 567.0, 1208.0 / 2],
#                       [0.0, 0.0, 1.0]])

intrinsic = np.array([[ 5.97615131e+02, -4.50346295e-01,  9.43058151e+02],
                      [ 0.00000000e+00,  5.97320118e+02,  5.66164841e+02],
                      [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
dist = np.array([-0.03800249,
                  0.04716124,
                 -0.04492046,
                  0.01374281])

new_intrinsic = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(intrinsic, dist, (1928, 1208), np.eye(3), fov_scale=1.5)

def find_vanishing_point(img):
    # Load the image and convert to grayscale
    # img = cv2.imread(img_path)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: gray = img.copy()

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

element = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))

def skeletonize(img):
    """Compute the skeleton of a binary image using morphological operations."""
    skel = np.zeros(img.shape, np.uint8)
    size = np.size(img)
    
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        done = (cv2.countNonZero(img) == 0)
        
    return skel


image_files = glob.glob("dataset" + "/*.jpg")
for image_file in image_files:
    # Load the image
    image = cv2.imread(image_file)

    # Convert the image to HSV format
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # lower_bound = np.array([12, 61,  0])  # Lower bound for green hue
    # upper_bound = np.array([166, 255, 129])  # Upper bound for green hue
    lower_bound = np.array([4, 114,  59])  # Lower bound for green hue
    upper_bound = np.array([177, 156, 100])  # Upper bound for green hue
    # Create a mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # (Optional) Bitwise-AND the mask and original image
    result = cv2.bitwise_and(image, image, mask=mask)
    print(mask.dtype)
    mask = cv2.fisheye.undistortImage(mask, intrinsic, dist, Knew=new_intrinsic)
    mask = cv2.dilate(mask, element)

    # Show the original image, mask, and result
    skeleton = skeletonize(mask.copy())

    cv2.imshow('skeleton', skeleton)
    cv2.imshow('Original Image', image)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
cv2.destroyAllWindows()


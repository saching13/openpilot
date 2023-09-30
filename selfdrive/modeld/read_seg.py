import cv2
import numpy as np
import glob
import supervision as sv


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

npy_files = glob.glob("dataset_segment/straight" + "/*.npy")
color_imgs = glob.glob("dataset_segment/straight" + "/*.jpg")
npy_files.sort()
color_imgs.sort()

for npy, im_file in zip(npy_files, color_imgs):
    mask = np.load(npy).astype(np.uint8) * 255
    im = cv2.imread(im_file)

    print(mask.shape)
    print(im.shape)
    # print(image_data.dtype)
    # sv.plot_image(image_data, size=(16, 16))
    # mask_clr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # res = np.hstack((im, mask_clr))
    rgb_size = im.shape
    mask_size = mask.shape
    offset  = 20
    croped_mask = mask[mask_size[0] - rgb_size[0]: mask_size[0], rgb_size[1] + offset: rgb_size[1]*2 + offset]
    # min_width = cv2.
    # mask 

    mask = cv2.dilate(croped_mask, element)
    sck = skeletonize(mask)
    sck = cv2.dilate(sck, element)

    cv2.imshow('croppedMask', croped_mask)
    cv2.imshow('sck', sck)
    cv2.imshow('mask', mask)
    cv2.imshow('img', im)

    cv2.waitKey(0)
cv2.destroyAllWindows()

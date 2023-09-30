from simplecnn import SimpleCNN
import torch
import glob
import numpy as np
import cv2


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


model = SimpleCNN()  # Assuming SimpleCNN is the model class
model.load_state_dict(torch.load('nav_model_hackathon_1.pth'))
model.eval()

base_dir = "dataset_segment_2/"
folders = ["straight", "left", "right"]

# data = []
# labels = []
res = {}
for label, folder in enumerate(folders):
    folderSubs =glob.glob(base_dir + folder + "/*")
    classCount = [0, 0, 0]
    print(f'Analyzing class {folder}')
    for subFolder in folderSubs:
        print(subFolder)
        npy_files = glob.glob(subFolder + "/*.npy")
        color_imgs = glob.glob(subFolder + "/*.jpg")
        npy_files.sort()
        color_imgs.sort()
        
        for npy_file, clr_file in zip(npy_files, color_imgs):
            mask = np.load(npy_file).astype(np.uint8) # they will be zero and ones
            im = cv2.imread(clr_file)
            
            rgb_size = im.shape
            mask_size = mask.shape

            offset  = 20
            croped_mask = mask[mask_size[0] - rgb_size[0]: mask_size[0], rgb_size[1] + offset: rgb_size[1]*2 + offset]
            
            mask = cv2.dilate(croped_mask, element)
            sck = skeletonize(mask)
            sck = cv2.dilate(sck, element)
            input_d = torch.from_numpy(np.expand_dims(sck, axis=0)).float()
            with torch.no_grad():  # Deactivate autograd, reduces memory usage and speeds up computations
                predictions = model(input_d)
            _, predicted_classes = predictions.max(1)  # Get the class with the highest probability
            classCount[predicted_classes] += 1
            print(f' Class {folder} predictions IS  -> {folders[predicted_classes]}')
            cv2.imshow('Path', sck*255)
            cv2.imshow('image', im)
            cv2.imshow('mask', mask*255)
            cv2.waitKey(0)

    res[folder] = classCount

print(f'Predictions -> \n {res}')

            




            # data.append(np.expand_dims(sck, axis=0))
            # labels.append(label)



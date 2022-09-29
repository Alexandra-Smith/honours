import cv2
import os
import numpy as np
from skimage.morphology import octagon
import math
from natsort import natsorted
from matplotlib import pyplot as plt
import nibabel as nib
from scipy import ndimage

# get all images from a certain folder with corresponding labels
def getImages(folder):
    files = os.listdir(folder)
    images = []
    p = os.path.join(folder)
    D = natsorted(os.listdir(p))
    for img in D:
        im = cv2.imread(os.path.join(p, img), cv2.IMREAD_GRAYSCALE)
        images.append(im)
    # images.pop(0)
    return images

def threshold(image):
    """
    Use Otsu's threshold along with morphological closing to get binary image
    """
    (T, threshInv) = cv2.threshold(image.astype(np.uint8), 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    closing = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, kernel)

    return closing

def background(im, mask):
    """
    Ensure image is on a complete (exact) black background, work with thresholded image as a mask.
    """
    r, c = im.shape
    image = np.ones((r, c), np.uint8)
    # move left to right
    for i in range(r):
        for j in range(c):
            if mask[i][j] == 0:
                image[i][j] = mask[i][j]
            elif mask[i][j] != 0:
                break   
    # move right to left
    for i in range(r):
        for j in range(c-1, 0, -1):
            if mask[i][j] == 0:
                image[i][j] = mask[i][j]
            elif mask[i][j] != 0:
                break 
    # go through whole image
    for i in range(r):
        for j in range(c):
            if image[i][j] == 1:
                image[i][j] = im[i][j]
    return image

def crop(image, seg):
    '''
    Crops image tightly around brain, leaves out unnecessary black spaces
    Use thresholded image to find coordinates (must have complete black background)
    '''
    r, c = image.shape
    # get thresholded image
    closing = threshold(image)

    # keep track of first row that contains image data
    top = None
    left = None
    right = None
    bottom = None
    # bottom space
    for i in range(r):
        for j in range(c):
            if closing[i][j] != 0:
                bottom = i
                break
    # top space
    for i in range(r-1, 0, -1):
        for j in range(c):
            if closing[i][j] != 0:
                top = i
                break
    # right space
    for i in range(c):
        for j in range(r):
            if closing[j][i] != 0:
                right = i
                break
    # left space
    for i in range(c-1, 0, -1):
        for j in range(r):
            if closing[j][i] != 0:
                left = i
                break
    # crop image
    cropped = image[top:bottom, left:right]
    tumour = seg[top:bottom, left:right]
    #print("[INFO] top: {}, bottom: {}, left: {}, right: {}".format(top, bottom, left, right))
    return cropped, tumour

def preprocess(image, size):
    """
    Combine all preprocessing steps in order to do all methods on one image at once.
    Includes: cropping, black background, making a standard size, skull stripping, formatting to jpg and saving output
    """
    r, c = image.shape
    mask = threshold(image)
    b = background(image, mask)
    cropped = crop(b)
    final = cv2.resize(cropped, (size, size))

    return cropped

def images(path, total, type, slice_number):
    # retrieve all images from folder
    train = [];
    for i in range(total):
        d = nib.load(path + str(i+1+330) + '/BraTs20_Training_' + str(i+1+330) + '_' + type + '.nii')
        train.append(d)
    # extract slices
    images = []
    for i in range(total):
        slice = train[i].dataobj[:, :, slice_number]
        images.append(ndimage.rotate(slice, -90))
    return images

def importing():
    '''importing nifti files and saving 
    slices as jpg format'''
    
    # path = '/Volumes/ALEX/Training/BraTS20_Training_'
    path = '/Users/alexandrasmith/Desktop/Honours Project/Databases/BraTS2020/Training/BraTS20_Training_'
    # total number of data files stored at specific path
    total = 39
    # slice number to be extracted
    slice_num = 100
    # type options: flair, t1, t1ce, t2 or seg
    train_images = images(path, total, 't2', slice_num)
    seg_images = images(path, total, 'seg', slice_num)
    # preprocess each image and corresponding segmented tumour
    X = []
    y = []
    for i in range(total):
        im, seg = crop(train_images[i], seg_images[i])
        # X.append(cv2.resize(im, (200, 200))) # make all images 200x200
        # y.append(cv2.resize(seg, (200, 200)))
        x = cv2.resize(im, (240, 240))
        y = cv2.resize(seg, (240, 240))
        plt.imsave('/Users/alexandrasmith/Desktop/Honours Project/Databases/BraTS2020/Extra/Slice' + str(slice_num) + '/T2/' + str(slice_num) + 't2_' + str(i+330) + '.jpg', x, cmap="gray")
        # plt.imsave('/Users/alexandrasmith/Desktop/Honours Project/Databases/BraTS2020/Extra/Slice' + str(slice_num) + '/Segmented/' + str(slice_num) + 'seg' + str(i+330) + '.jpg', y, cmap="gray")

def Seg(folder):
    '''convert the segmented images into binary format
    such that the whole tumour is represented by white pixels'''
    path = '/Users/alexandrasmith/Desktop/Honours Project/Databases/BraTS2020/' + folder
    # get all data
    seg = getImages(path)
    # first change segmented images to binary
    # ground truth images only contain one type of tumour
    for I in range(len(seg)):
        img = seg[I]
        r, thresh = cv2.threshold(img, 18, 255, cv2.THRESH_BINARY)
        plt.imsave('/Users/alexandrasmith/Desktop/Honours Project/Databases/BraTS2020/Segmented/seg_' + str(I) + '.jpg', thresh, cmap="gray")
        # plt.imsave('/Users/alexandrasmith/Desktop/S/seg_' + str(I) + '.jpg', thresh, cmap="gray")

def Filter(folder):
    '''find segmented images that contain no tumour 
    so that those files can be removed from training set'''

    path = '/Users/alexandrasmith/Desktop/Honours Project/Databases/BraTS2020/' + folder
    # get all data
    seg = getImages(path)
    # plt.figure(); plt.imshow(seg[368], cmap="gray"); plt.show()

    list = []
    for i in range(len(seg)-1):
        img = seg[i]
        white = False
        for a in range(240):
            for b in range(240):
                if img[a, b] == 255:
                    white = True
        if white == False:
            list.append(i)
    return list

def main():

    # check if tumour exists in segmented image
    folder = 'Extra/Slice100/Segmented'

    l = Filter(folder)
    print(l)

if __name__ == "__main__":
    main()
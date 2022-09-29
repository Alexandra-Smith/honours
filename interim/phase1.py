#import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
from IPython.display import display
import os
import skimage.color
from skimage import filters
import skimage.io
import skimage.viewer


# get all images from a certain folder
def getImages(folder):
    files = os.listdir(folder)
    images = []
    for file in files:
        filePath = os.path.abspath(os.path.join(folder, file))
        try:
            fp = open(filePath, "rb")
            im = Image.open(fp)
            images.append(im)
            im.load()
            fp.close()
        except:
            print("Invalid image: %s" % (filePath,))
    return images


def main():
    path = '/Users/alexandrasmith/Desktop/Project/Databases/Dataset 1'

    # get all healthy brain images
    healthy = getImages(path + '/n')

    # PREPROCESSING

    # for i in range(len(healthy)):
    #for img in healthy:
        # ensure greyscale

        # crop

        # apply Gaussian filter

        # output - save as .jpg file
        #img.save('.jpg')


    # one image to test
    im = Image.open(path + '/n/no_3.jpg')

    # applying grayscale method
    grey = ImageOps.grayscale(im)
    g = np.asarray(grey)

    #im.thumbnail((500, 500))
    #im.save(path + '/th.jpg')

    # make histogram
    plt.hist(g.ravel(),256)
    #plt.show()

    # Gaussian filter to remove noise
    blur = grey.filter(ImageFilter.GaussianBlur(2))
    #blur.show()

    t = skimage.filters.threshold_otsu(blur)
    mask = blur > t
    mask.show()

    sel = np.zeros_like(image)
    sel[mask] = image[mask]

    #sel.show()


    # cropping images
    # (left, upper, right, lower) = (left, upper, (left+width), (upper+height))
    #im_crop = im.crop((left, upper, right, lower))

if __name__ == "__main__":
    main()
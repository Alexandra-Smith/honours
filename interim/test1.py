'''
Set up to test all functions on one image before applying to entire dataset.
Practice pre-processing.
'''

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from skimage.morphology import octagon
from sklearn import metrics
from scipy.spatial.distance import cdist
from natsort import natsorted
from sklearn import metrics
from scipy import ndimage
from scipy.ndimage.filters import median_filter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import skimage.measure as measures

# get all images from a certain folder with corresponding labels
def getImages(folder, category):
    files = os.listdir(folder)
    images = []
    labels = []
    p = os.path.join(folder, category)
    D = natsorted(os.listdir(p))
    for img in D:
        im = cv2.imread(os.path.join(p, img), cv2.IMREAD_GRAYSCALE)
        images.append(im)
        labels.append(category)
    images.pop(0)
    labels.pop(0)
    return images, labels

def crop(image):
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
    #print("[INFO] top: {}, bottom: {}, left: {}, right: {}".format(top, bottom, left, right))
    return cropped

def threshold(image):
    """
    Use Otsu's threshold along with morphological closing to get binary image
    """
    (T, threshInv) = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
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

def preprocess(image):
    """
    Combine all preprocessing steps in order to do all methods on one image at once.
    Includes: cropping, black background, making a standard size, skull stripping, formatting to jpg and saving output
    """
    r, c = image.shape
    mask = threshold(image)
    b = background(image, mask)
    strip = skull_strip(b)
    cropped = crop(strip)
    final = cv2.resize(cropped, (512, 512))

    return final

def skull_strip(image):
    """
    Function to get rid of skull in image and isolate the brain tissue.
    Uses mask from thresholding process as beginning image.
    """

    # Gaussian filter
    blur = cv2.GaussianBlur(image, (5,5), 0.7)

    # Thresholding using Otsu
    (T, thresh) = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Morphological operators
    oct = octagon(3, 3)
    erosion = cv2.erode(thresh, oct, iterations = 1)
    erosion2 = cv2.erode(erosion, oct, iterations = 1)
    dilation = cv2.dilate(erosion2, oct, iterations = 1)
    dilation2 = cv2.dilate(dilation, oct, iterations = 1)
    
    # get region boundaries (contours)
    contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # NOTE: BGR not RGB colour format
    image_col = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # draw contours
    c = image_col.copy()
    cv2.drawContours(c, contours, -1, (219, 112, 147), 2)
    max_contour = max(contours, key = cv2.contourArea)
    outline = image_col.copy()
    cv2.drawContours(outline, [max_contour], 0, (219, 112, 147), 2)

    # create mask
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [max_contour], 0, 255, -1)

    # final skull stripped image
    stripped = cv2.bitwise_and(image, image, mask = mask)

    ims = np.hstack((cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR), cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), cv2.cvtColor(dilation2, cv2.COLOR_GRAY2BGR), c, outline, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), cv2.cvtColor(stripped, cv2.COLOR_GRAY2BGR)))

    return stripped


def split(data):
    '''
    Split data into training and test data
    '''
    train, test = train_test_split(all_data, train_size=0.95, random_state=0)
    return train, test

def kmeans(image, k):
    # flatten image
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # perform clustering 
    _, labels, centres = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centres = np.uint8(centres)
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centres[labels.flatten()]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image

def hsv_blues(image):
    ''' Colour segmentation/filtering for blue/green channels, using hsv colour image '''

    # blue range
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    # green range
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])

    kernel = np.ones((9, 9), np.uint8)
    blue_mask = cv2.inRange(image, lower_blue, upper_blue)
    invert_blue = cv2.bitwise_not(blue_mask)
    green_mask = cv2.inRange(image, lower_green, upper_green)

    closing_blue = cv2.morphologyEx(invert_blue, cv2.MORPH_CLOSE, kernel)
    inverted_blue = cv2.bitwise_not(closing_blue)

    closing_green = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    blue = np.hstack((blue_mask, inverted_blue))
    green = np.hstack((green_mask, closing_green))

    final_mask = closing_green + inverted_blue

    inv_final = cv2.bitwise_not(final_mask)
    final = cv2.morphologyEx(inv_final, cv2.MORPH_OPEN, kernel)

    return final

def hsv_reds(image):
    ''' Colour segmentation/filtering for red/orange/yellow channels, using hsv colour image '''

    # yellow range
    lower_yellow = np.array([25, 50, 50])
    upper_yellow = np.array([35,255,255])
    # orange range
    lower_orange = np.array([10, 50, 50])
    upper_orange = np.array([24, 255, 255])
    # red first range
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([9, 255, 255])
    # red second range
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    yellow_mask = cv2.inRange(image, lower_yellow, upper_yellow)
    orange_mask = cv2.inRange(image, lower_orange, upper_orange)
    red_mask1 = cv2.inRange(image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(image, lower_red2, upper_red2)

    kernel = np.ones((9, 9), np.uint8)
    closing_yellow = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    closing_orange = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
    closing_red1 = cv2.morphologyEx(red_mask1, cv2.MORPH_CLOSE, kernel)
    closing_red2 = cv2.morphologyEx(red_mask2, cv2.MORPH_CLOSE, kernel)

    final_mask = closing_yellow + closing_orange + closing_red1 + closing_red2

    final = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

    return final

def clustering(image, threshold):
    ''' 
    Makes use of preprocessed image
    Includes: colour transformation, clustering technique, hsv transformation, colour filtering
    '''

    # pseudo colour transformation
    colour = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    # kmeans on colour image
    seg3 = kmeans(colour, 3)
    # convert to hsv image
    hsv = cv2.cvtColor(seg3, cv2.COLOR_BGR2HSV)
    reds_mask = hsv_reds(hsv)
    blues_mask = hsv_blues(hsv)
    m = reds_mask + blues_mask

    kernel = np.ones((21, 21), np.uint8)
    msk = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    images = np.hstack((colour, seg3, cv2.cvtColor(m, cv2.COLOR_GRAY2BGR), cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)))
    cv2.imshow("images", images)
    cv2.waitKey()

    cv2.imwrite('/Users/alexandrasmith/Desktop/wrong_colour.jpg', colour)
    cv2.imwrite('/Users/alexandrasmith/Desktop/wrong_seg.jpg', seg3)
    #cv2.imwrite('/Users/alexandrasmith/Desktop/wrong_mask5.jpg', m)

    label = 0
    tt = 0
    # check if red colour exists in image
    for i in range(msk.shape[0]):
        for j in range(msk.shape[1]):
            if msk[i][j] != 0:
                #print("DETECTED: possible tumour")
                label = 1
                #tt += 1
                break
    
    # r, c = image.shape
    # total = tt/(r*c)*100

    # if total > threshold:
    #     label = 1

    return label

def components(mask):
    ''' 
    Uses connected components to identify regions.
    Takes in the mask image obtained from colour filtering
    '''

    labeled_image, nums = measures.label(mask, connectivity=2, return_num=True)
    region_features = measures.regionprops(labeled_image)
    im = np.uint8(labeled_image)

    # image with very large number of components should not have a tumour
    if nums > 100:
        label = 0
    else:
        label = 1

    object_centres = [objf["centroid"] for objf in region_features]
    object_areas = [objf["area"] for objf in region_features]

    # ignore components with small areas
    for i in range(nums-1, -1, -1):
        area = object_areas[i]
        if area < 50:
            del object_areas[i]
            del object_centres[i]
    
    #print(nums)
    #print(np.asarray(object_centres))

    # m1 = np.mean(object_centres, axis=0)[0]
    # m2 = np.mean(object_centres, axis=0)[1]
    # # calculate variance in x axis
    # v1 = np.var(object_centres, axis=0)[0]
    # # calculate variance in y direction
    # v2 = np.var(object_centres, axis=0)[1]

    # based on distances 
    # see if white regions are stretched far over image
    max_distx = 0
    max_disty = 0
    # for c in object_centres:
    #     for d in object_centres:
    #         dist = np.linalg.norm(np.asarray(c) - np.asarray(d))
    #         if dist > max_dist:
    #             max_dist = dist
    for c in object_centres:
        x = c[0]
        y = c[1]
        for d in object_centres:
            xx = d[0]
            yy = d[1]
            distx = x - xx
            disty = y - yy
            if distx > max_distx:
                max_distx = distx
            if disty > max_disty:
                max_disty = disty
    
    #print(max_distx, max_disty)

    # if label == 1:
    #     if max_distx > 250 and max_disty > 250:
    #         label = 0
    #     else:
    #         label = 1

    # print(m1, m2)
    # print(np.sqrt(v1), np.sqrt(v2))
    # cv2.imshow("mask", mask)
    # cv2.imshow("regions", im)
    # cv2.waitKey()

    return label

def predict(all_images, threshold):
    # get predicted labels
    predicted_labels = []
    # apply to all training images
    for i in range(len(all_images)):
        img = all_images[i]
        pred_l = clustering(img, threshold)
        # if pred_l == 1:
        #     pred_l = components(mask)
        predicted_labels.append(pred_l)
    
    return predicted_labels

def get_accuracy(true_labels, predicted_labels):
    # check accuracy
    count = 0
    total = len(true_labels)
    for i in range(total):
        if predicted_labels[i] == true_labels[i]:
            count += 1
        # else:
        #     print(i)
    per = count/total*100
    #print("Accuracy " + str(per) + '%')

    return per

def cm(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    cmd_obj = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['healthy', 'tumour'])
    cmd_obj.plot(include_values=True, cmap='Blues')
    cmd_obj.ax_.set(
                title='Confusion matrix', 
                xlabel='Predicted', 
                ylabel='Actual')
    plt.show()

def main():
    # display figure of image with scale and save
    # plt.imshow(im ,cmap="gray")
    # plt.show()
    # plt.savefig('/Users/alexandrasmith/Desktop/Honours Project/Assessments/Interim Report/original_scale.jpg')

    #P = '/Users/alexandrasmith/Desktop/Honours Project/Databases/Final set'
    P = '/Users/alexandrasmith/Desktop/Honours Project/Databases/Dataset 1/Processed'
    # ALL PREPROCESSED TRAINING DATA AVAILABLE
    X_healthy, y_healthy = getImages(P, "no")
    X_tumour, y_tumour = getImages(P, "yes")
    all_data = X_healthy + X_tumour
    all_labels = y_healthy + y_tumour
    true_labels = []
    for i in range(len(all_labels)):
        if all_labels[i] == 'no':
            true_labels.append(0)
        else:
            true_labels.append(1)
    
    # get single image
    #img = X_healthy[71]
    img = X_tumour[26]
    r, c = img.shape

    clustering(img, 0)
    
    # p = predict(all_data, 0)
    # acc = get_accuracy(true_labels, p)
    # cm(true_labels, p)
    # print("Accuracy: " + str(acc) + '%')

    # fpr, tpr, thresholds = metrics.roc_curve(true_labels, p)
    # score = metrics.roc_auc_score(true_labels, p)
    # print(score)
    # print(thresholds)

    # plt.plot(fpr, tpr, 'k--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC curve')
    # plt.show()

    # # set threshold
    # t = 25

    # # calculating FRR
    # # calculate using healthy set, check how many healthy images get 'rejected'/classified as tumour
    # total1 = len(y_healthy)
    # rej = 0
    # p1 = predict(X_healthy, t)
    # for i in p1:
    #     if i == 1:
    #         rej += 1
    # frr = rej/total1*100
        
    # # calculating FAR
    # # calculate on tumour test set, check how many are imposters (tumours), ie how many tumours are predicted as healthy/how many tumours pass as healthy
    # total2 = len(y_tumour)
    # imp = 0
    # p2 = predict(X_tumour, t)
    # # calculate how many tumours get labelled as healthy
    # for i in p2:
    #     if i == 0:
    #         imp += 1
    # far = imp/total2*100

    # predicted = p1 + p2
    # acc = get_accuracy(true_labels, predicted)
    
    # print("Threshold: " + str(t))
    # print("FRR: " + str(frr))
    # print("FAR: " + str(far))
    # print("Accuracy " + str(acc) + '%')

    # # calculate the EER
    # for i in range(100):
    #     a = frr[i]
    #     b = far[i]
    #     if a == b:
    #             EER = a
    #             print('EER = ', i)

    # fig, ax = plt.subplots()
    # ax.plot(threshold, far, 'b--', label='FAR')
    # ax.plot(threshold, frr, 'g--', label='FRR')
    # plt.xlabel('Threshold')
    # plt.plot(15, EER, 'ro', label='EER') 
    # legend = ax.legend(loc='upper center', shadow=True, fontsize='large')

    # # Put a nicer background color on the legend.
    # legend.get_frame().set_facecolor('gray')
    # plt.show()

    # # apply to all training images
    # for i in range(len(X_healthy)):
    #     img = X_healthy[i]
    #     new = preprocess(img)
    #     cv2.imwrite('/Users/alexandrasmith/Desktop/Honours Project/Databases/Dataset 1/Processed/no/no' + str(i) + '.jpg', new)

if __name__ == "__main__":
    main()
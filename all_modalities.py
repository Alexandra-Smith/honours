import cv2
import os
import numpy as np
import math
from sklearn import metrics as m
from natsort import natsorted
from matplotlib import pyplot as plt
import nibabel as nib
from scipy import ndimage
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout, LeakyReLU
from sklearn.model_selection import train_test_split
import visualkeras
from collections import defaultdict
from sklearn.model_selection import KFold

def getImages(folder):
    files = os.listdir(folder)
    images = []
    p = os.path.join(folder)
    D = natsorted(os.listdir(p))
    names = [f for f in D if not f.startswith('.')]
    for img in names:
        im = cv2.imread(os.path.join(p, img), cv2.IMREAD_GRAYSCALE)
        images.append(im)
    return images

path = '/Users/alexandrasmith/Desktop/Honours Project/Databases/Phase2/'

flair = getImages(path + 'Flair')
t1 = getImages(path + 'T1')
t1ce = getImages(path + 'T1ce')
t2 = getImages(path + 'T2')
seg = getImages(path + 'Segmented')

# creating array of all modalities
total = len(seg)
# X is a 1499x240x240x4 array - training data containing all modalities
X4 = np.zeros((total, 240, 240, 4))
for i in range(total):
    img1 = flair[i]; img2 = t1[i]; img3 = t1ce[i]; img4 = t2[i]
    X4[i, :, :, 0] = img1; X4[i, :, :, 1] = img2; X4[i, :, :, 2] = img3; X4[i, :, :, 3] = img4

# create data sets used for training (using ALL modalities together)
def get_data(X, patch_size, number_of_ims, num_strides):
    '''Create data sets used for training
    Returns:    array of image patches
                image (pixel) labels'''
    idx = np.int(np.floor(patch_size/2))
    # strides = 1 -> 42436 patches per image (VALID padding)
    # strides = 2 -> 10609 patches per image
    # stides = 3 -> 4761 patches per image
    # stides = 4 -> 2704 patches per image
    # shape of matrix of total patches = [number_of_images_used*number_patches_per_image, 35, 35]
    all_patches = np.zeros(((number_of_ims*1225), 35, 35, 4))
    all_patches_labels = np.zeros((number_of_ims*1225, 35, 35, 1))
    for i in range(number_of_ims):
        F = np.reshape(X[i], (1, 240, 240, 4))
        S = np.reshape(seg[i], (1, 240, 240, 1))
        P = tf.image.extract_patches(images=F, sizes=[1, patch_size, patch_size, 1], strides=[1, num_strides, num_strides, 1], rates=[1, 1, 1, 1], padding='VALID')
        P_seg = tf.image.extract_patches(images=S, sizes=[1, patch_size, patch_size, 1], strides=[1, num_strides, num_strides, 1], rates=[1, 1, 1, 1], padding='VALID')
        p = P.numpy(); p_seg = P_seg.numpy()
        sh = p.shape; num_patches = np.int((sh[1]**2))
        # print(num_patches)
        # get numpy array of size (number_patches, patch_size, patch_size)
        patches = np.reshape(p, (num_patches, patch_size, patch_size, 4)); y_patches = np.reshape(p_seg, (num_patches, patch_size, patch_size, 1))
        for k in range(num_patches):
            all_patches[k, :, :] = patches[k, :, :]
            all_patches_labels[k, :, :] = y_patches[k, :, :]

    # get pixel labels for centre pixel of each patch: label 0 or 1 corresponding to two class labels
    # 0 == healthy/background and 1 == tumorous
    all_pix_labels = []
    for patch in all_patches_labels:
        i = patch[idx][idx]
        if i == 255:
            all_pix_labels.append(1)
        else:
            all_pix_labels.append(0)
    return all_patches, all_pix_labels

def seg_image(model, X, seg, patch_size, im_num, thresh):
    '''Predict labels and segment for one given image'''
    idx = np.int(np.floor(patch_size/2))
    test_ = np.reshape(X4[im_num], (1, 240, 240, 4)); seg_ = np.reshape(seg[im_num], (1, 240, 240, 1))
    # extract image patches
    P_test = tf.image.extract_patches(images=test_, sizes=[1, patch_size, patch_size, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    P_seg_test = tf.image.extract_patches(images=seg_, sizes=[1, patch_size, patch_size, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    # convert to numpy array
    p_test = P_test.numpy(); p_seg_test = P_seg_test.numpy()
    # reshape(number_patches, patch_size, patch_size)
    test_patches = np.reshape(p_test, (57600, patch_size, patch_size, 4)); test_y_patches = np.reshape(p_seg_test, (57600, patch_size, patch_size, 1))
    test_pix = []
    for patch in test_y_patches:
        i = patch[idx][idx]
        if i == 255:
            test_pix.append(1)
        else:
            test_pix.append(0)

    # normalise and one-hot encoding
    test_patches = test_patches/255; test_patches = np.reshape(test_patches, (test_patches.shape[0], test_patches.shape[1], test_patches.shape[2], 1))
    test_pix_oh = tf.one_hot(test_pix, depth=2)

    Loss, Acc = model.evaluate(test_patches, test_pix_oh)
    labels_ = model.predict(test_patches)
    # apply threshold for pixel to be classified as tumorous
    T = np.where(labels_ < thresh, 0, labels_)
    ll = np.argmax(T, axis=-1) 
    test_seg_image = np.reshape(ll, (240, 240))
    acc = get_measures(test_pix, ll)
    print(f'Loss: {round(Loss,2)}')
    # plot segmentation result (if one image given)
    # plt.figure(figsize=(20, 10))
    # plt.subplot(131); plt.imshow(X4[im_num, :, :, 0], cmap="gray"); plt.axis('off'); plt.title("Original flair image")
    # plt.subplot(132); plt.imshow(seg[im_num], cmap="gray"); plt.axis('off'); plt.title("Ground truth")
    # plt.subplot(133); plt.imshow(test_seg_image, cmap="gray"); plt.axis('off'); plt.title("Predicted. Accuracy: " + str(acc) + ". Threshold: " + str(thresh))
    # plt.tight_layout(); plt.show()

    return test_seg_image

def split(X, y):
    ''' Split all patches into training and
    test data, normalise and one hot encoding done to
    ensure data in correct format for training CNN.'''
    # split into training and test (20%) sets
    X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # get validation data
    X_train, X_val, y_train, y_val = train_test_split(X_t, y_t, test_size=0.125, random_state=0)
    # normalizing the pixel values and reshape data
    X_train = X_train/255; X_test = X_test/255; X_val = X_val/255
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 4))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 4))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2], 4))
    # one hot encoding
    y_train = tf.one_hot(y_train, depth=2); y_test = tf.one_hot(y_test, depth=2); y_val = tf.one_hot(y_val, depth=2)
    return X_train, y_train, X_val, y_val, X_test, y_test

def train(X_train, y_train, X_val, y_val):
    ''' Configure and train CNN'''
    model = models.Sequential([
        Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding = 'same', input_shape=(35, 35, 4)),
        # MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(units=128, activation='relu'), # fully connected layer
        Dropout(0.5),
        # Flatten(),
        Dense(units=2, activation='softmax'), # output layer
    ])
    # model = models.Sequential([
    # Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(35, 35, 4),padding='same'),
    # LeakyReLU(alpha=0.1),
    # MaxPool2D((2, 2),padding='same'),
    # Conv2D(64, (3, 3), activation='linear',padding='same'),
    # LeakyReLU(alpha=0.1),
    # MaxPool2D(pool_size=(2, 2),padding='same'),
    # Conv2D(128, (3, 3), activation='linear',padding='same'),
    # LeakyReLU(alpha=0.1),
    # MaxPool2D(pool_size=(2, 2),padding='same'),
    # Flatten(),
    # Dense(128, activation='linear'),
    # LeakyReLU(alpha=0.1),
    # Dense(2, activation='softmax'),
    # ])

    # display model architecture
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, np.array(y_train), batch_size=64, epochs=15, validation_data=(X_val, np.array(y_val)), shuffle=True)

    return model, history

def evaluate_model(model, history, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], 'm--', label='validation')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], 'm--', label='validation')
    plt.legend()
    return loss, accuracy

def get_measures(true_labels, pred_labels):
    # calculate other statistical measures
    cm = m.confusion_matrix(true_labels, pred_labels)

    tp = cm[1][1]; tn = cm[0][0]; fp = cm[0][1]; fn = cm[1][0]
    conf_accuracy = (float (tp+tn) / float(tp + tn + fp + fn)) # calculate accuracy
    conf_misclassification = 1 - conf_accuracy # calculate mis-classification
    conf_sensitivity = (tp / float(tp + fn)) # calculate the sensitivity
    conf_specificity = (tn / float(tn + fp)) # calculate the specificity
    conf_precision = (tp / float(tp + fp)) # calculate precision
    dsc = (2*tp) / (fp + 2*tp + fn) # calculate dice similarity coefficient

    print(f'Accuracy: {round(conf_accuracy,2)}') 
    print(f'Mis-Classification: {round(conf_misclassification,2)}') 
    print(f'Sensitivity: {round(conf_sensitivity,2)}') 
    print(f'Specificity: {round(conf_specificity,2)}') 
    print(f'Precision: {round(conf_precision,2)}')
    print(f'DSC: {round(dsc,2)}')

    return conf_accuracy

def acc(model, X_test, y_test, thresh):
    '''Manually calculate accuracy for patches 
    and given pixel classes'''
    # checking accuracy manually
    pred_prob = model.predict(X_test)
    # apply threshold
    T = np.where(pred_prob < thresh, 0, pred_prob)
    pred_labels = np.argmax(T, axis=-1)
    # convert tensor test labels into vector
    y_labels = []
    for i in range(len(pred_labels)):
        m = y_test[i]
        if m[0] == 1:
            y_labels.append(0)
        elif m[1] == 1:
            y_labels.append(1)
    count = 0
    for i in range(len(pred_labels)):
        if pred_labels[i] == y_labels[i]:
            count += 1
    per = count/len(pred_labels)
    acc = get_measures(y_labels, pred_labels)
    return per
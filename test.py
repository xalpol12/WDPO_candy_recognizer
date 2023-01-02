from pathlib import Path
from typing import Dict

import click
import cv2
import numpy as np
from tqdm import tqdm


def saturate_image(img, alpha, beta, gamma):
    new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    #gamma correction of the oversaturated regions:
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i/255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(new_image, lookUpTable)
    return res


def threshold(img, value, mode):
    _, thresh1 = cv2.threshold(img, value, 255, mode)
    return thresh1


def remove_shadows(img):
    dilated = cv2.dilate(img, np.ones((9,9), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(img, bg)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return norm


def saturation_test(imgpath):
    def empty_callback(value):
        pass

    image = cv2.imread(imgpath)
    if image is None:
        print('Could not open or find the image: ', imgpath)
        exit(0)
    resize_percent = 0.3
    width = int(image.shape[1] * resize_percent)
    height = int(image.shape[0] * resize_percent)
    image = cv2.resize(image, (width, height), interpolation = cv2.INTER_NEAREST)
    cv2.namedWindow('image1')
    cv2.createTrackbar('alpha', 'image1', 1, 30, empty_callback)
    cv2.createTrackbar('beta', 'image1', 0, 100, empty_callback)
    cv2.createTrackbar('gamma', 'image1', 0, 250, empty_callback)

    while True:
        alpha = cv2.getTrackbarPos('alpha', 'image1') / 10
        beta = cv2.getTrackbarPos('beta', 'image1')
        gamma = cv2.getTrackbarPos('gamma', 'image1') /10

        cv2.imshow('image1', saturate_image(image, alpha, beta, gamma))

        key_code = cv2.waitKey(10)
        if key_code == 27:
            thresholding_test('', saturate_image(image, alpha, beta, gamma))


def thresholding_test(imgpath, picture=np.array([])):
    def empty_callback(value):
        pass

    if picture.any() != 0:
        img = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.imread(imgpath, 0)
        if img is None:
            print('Could not open or find the image: ', imgpath)
            exit(0)
    cv2.namedWindow('image')

    cv2.createTrackbar('Threshold', 'image', 0, 255, empty_callback)
    cv2.createTrackbar('Tryb', 'image', 0, 4, empty_callback)

    while True:
        trsh = cv2.getTrackbarPos('Threshold', 'image')
        mode = cv2.getTrackbarPos('Tryb', 'image')
        if mode == 0:
            tryb_progowania = cv2.THRESH_BINARY
        elif mode == 1:
            tryb_progowania = cv2.THRESH_BINARY_INV
        elif mode == 2:
            tryb_progowania = cv2.THRESH_TRUNC
        elif mode == 3:
            tryb_progowania = cv2.THRESH_TOZERO
        elif mode == 4:
            tryb_progowania = cv2.THRESH_TOZERO_INV

        image = threshold(img, trsh, tryb_progowania)
        cv2.imshow("image", image)
        key_code = cv2.waitKey(10)
        if key_code == 27:
            cv2.imshow('new window', remove_shadows(threshold(img, trsh, tryb_progowania)))



if __name__ == '__main__':
    saturation_test("data/00.jpg")
    # thresholding_test("data/00.jpg")
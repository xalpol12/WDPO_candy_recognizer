import os
import cv2
import numpy as np
import json

#HSV values for different colors:
#green low_hsv = [44, 121, 35]  high_hsv = [86, 255, 229]
#purple (do sprawdzenia) low_hsv = [106, 52, 49] high_hsv = [178, 174, 115]
#red low_hsv = [106, 155, 62] high_hsv = [123, 255, 252]
#yellow low_hsv = [88, 163, 88] high_hsv = [115, 239, 246]


max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_trackbar_name = "Trackbars"
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

#on_change trackbar behaviour:
def on_low_H_thresh_trackbar(val):
    global low_H, high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_trackbar_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H, high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_trackbar_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S, high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_trackbar_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S, high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_trackbar_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V, high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_trackbar_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V, high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_trackbar_name, high_V)


def get_images_dir():
    image_dir = r'C:\Users\dawidexpompa2000\Desktop\Srudia\PO5_WDPO\Laby\WDPO_candy_recognizer\data'
    image_dir = image_dir.replace('\\', '/')  #change backslashes to forward slashes
    return image_dir


def get_next_image(index, image_dir, image_list, low_HSV, high_HSV):
    next_image = cv2.imread(os.path.join(image_dir, image_list[index]))
    next_image_processed = process_image(next_image, low_HSV, high_HSV)
    mask = create_mask(next_image_processed)
    return next_image, mask


def process_image(image, low_HSV, high_HSV):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # blurred = cv2.medianBlur(gray, 7)
    filtered = cv2.bilateralFilter(gray, 9, 20, 10)
    threshold = cv2.inRange(filtered, low_HSV, high_HSV)
    return threshold


def resize_image(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    image = cv2.resize(image, (width, height), interpolation = cv2.INTER_NEAREST)
    return image


def create_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_open = cv2.morphologyEx(mask_close,cv2.MORPH_OPEN, kernel)
    return mask_open


def get_contours(mask_open):
    contours, _ = cv2.findContours(mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def blend_with_mask(image, mask):
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(image, 1, mask_color, 1, 0)
    return blended
# 3072 x 4080 = 12 533 760
# 55 x 53 = 2 915


def get_candy_count(contours, image):
    height = 0; width = 0
    height, width = image.shape[:2]
    candy_count = 0
    for contour in contours:
        if cv2.contourArea(contour) > 0.0003 * height * width:
            candy_count += 1
    return candy_count


def main():
    global low_H, low_S, low_V, high_H, high_S, high_V, window_trackbar_name

    image_dir = get_images_dir()
    image_list = os.listdir(image_dir)
    current_image_index = 0
    image, mask = get_next_image(current_image_index, image_dir, image_list,
                                 np.array([low_H, low_S, low_V]), np.array([high_H, high_S, high_V]))

    #json import
    with open('sources/the_truth.json') as user_file:
        reference_json = json.load(user_file)
        # print(reference_json['00.jpg']['red'])


    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_trackbar_name, cv2.WINDOW_NORMAL)

    # create trackbars:
    cv2.createTrackbar(low_H_name, window_trackbar_name, low_H, max_value_H, on_low_H_thresh_trackbar)
    cv2.createTrackbar(high_H_name, window_trackbar_name, high_H, max_value_H, on_high_H_thresh_trackbar)
    cv2.createTrackbar(low_S_name, window_trackbar_name, low_S, max_value, on_low_S_thresh_trackbar)
    cv2.createTrackbar(high_S_name, window_trackbar_name, high_S, max_value, on_high_S_thresh_trackbar)
    cv2.createTrackbar(low_V_name, window_trackbar_name, low_V, max_value, on_low_V_thresh_trackbar)
    cv2.createTrackbar(high_V_name, window_trackbar_name, high_V, max_value, on_high_V_thresh_trackbar)

    while True:

        in_range = process_image(image, np.array([low_H, low_S, low_V]), np.array([high_H, high_S, high_V]))
        blended_contours = blend_with_mask(image, mask)
        cv2.drawContours(blended_contours, get_contours(mask), -1, (255,0,0), 3)

        cv2.imshow(window_trackbar_name, in_range)
        cv2.imshow('image', blended_contours)

        # switching images using 'w' and 'q'
        key = cv2.waitKey(10)
        if key == ord('q'):
            current_image_index -= 1
            if current_image_index < 0:
                current_image_index = len(image_list) - 1
            image, mask = get_next_image(current_image_index, image_dir, image_list,
                                         np.array([low_H, low_S, low_V]), np.array([high_H, high_S, high_V]))
        elif key == ord('w'):
            current_image_index += 1
            if current_image_index >= len(image_list):
                current_image_index = 0
            image, mask = get_next_image(current_image_index, image_dir, image_list,
                                         np.array([low_H, low_S, low_V]), np.array([high_H, high_S, high_V]))
        elif key == 27:
            break

    # Destroy the window
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
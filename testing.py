import os
import cv2
import numpy as np

#HSV values for different colors:


def get_images_dir():
    image_dir = r'C:\Users\dawidexpompa2000\Desktop\Srudia\PO5_WDPO\Laby\WDPO_candy_recognizer\data'
    image_dir = image_dir.replace('\\', '/')  #change backslashes to forward slashes
    return image_dir


def get_next_image(index, image_dir, image_list, low_HSV, high_HSV):
    next_image = cv2.imread(os.path.join(image_dir, image_list[index]))
    next_image_processed = process_image(next_image, low_HSV, high_HSV)
    mask = create_mask(next_image_processed)
    contours = get_contours(mask)
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
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


def get_candy_count(contours):
    candy_count = 0
    for i in range(len(contours)):
        candy_count += 1
    return candy_count




def main():
    def empty_callback(value):
        pass

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
    low_HSV = np.array([180, 255, 255])
    high_HSV = np.array([0, 0, 0])

    image_dir = get_images_dir()
    image_list = os.listdir(image_dir)
    current_image_index = 0
    image, mask = get_next_image(current_image_index, image_dir, image_list,
                                 low_HSV, high_HSV)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_trackbar_name, cv2.WINDOW_NORMAL)

    # create trackbars:
    cv2.createTrackbar(low_H_name, window_trackbar_name, low_H, max_value_H, empty_callback)
    cv2.createTrackbar(high_H_name, window_trackbar_name, high_H, max_value_H, empty_callback)
    cv2.createTrackbar(low_S_name, window_trackbar_name, low_S, max_value, empty_callback)
    cv2.createTrackbar(high_S_name, window_trackbar_name, high_S, max_value, empty_callback)
    cv2.createTrackbar(low_V_name, window_trackbar_name, low_V, max_value, empty_callback)
    cv2.createTrackbar(high_V_name, window_trackbar_name, high_V, max_value, empty_callback)

    while True:
        low_HSV = np.array([cv2.getTrackbarPos(low_H_name, window_trackbar_name),
                            cv2.getTrackbarPos(low_S_name, window_trackbar_name),
                            cv2.getTrackbarPos(low_V_name, window_trackbar_name)])

        high_HSV = np.array([cv2.getTrackbarPos(high_H_name, window_trackbar_name),
                             cv2.getTrackbarPos(high_S_name, window_trackbar_name),
                             cv2.getTrackbarPos(high_V_name, window_trackbar_name)])

        in_range = process_image(image, low_HSV, high_HSV)
        cv2.imshow(window_trackbar_name, in_range)
        blended_contours = blend_with_mask(image, mask)
        cv2.drawContours(blended_contours, get_contours(mask), -1, (255,0,0), 3)
        cv2.imshow('image', blended_contours)

        key = cv2.waitKey(0)
        if key == ord('q'):
            current_image_index -= 1
            if current_image_index < 0:
                current_image_index = len(image_list) - 1
            image, mask = get_next_image(current_image_index, image_dir, image_list, low_HSV, high_HSV)
        elif key == ord('w'):
            current_image_index += 1
            if current_image_index >= len(image_list):
                current_image_index = 0
            image, mask = get_next_image(current_image_index, image_dir, image_list, low_HSV, high_HSV)
        else:
            break

    # Destroy the window
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
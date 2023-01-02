import json
from pathlib import Path
from typing import Dict

import click
import cv2
import numpy as np
from tqdm import tqdm


# def saturate_image(img, alpha, beta, gamma):
#     new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
#     #gamma correction of the oversaturated regions:
#     lookUpTable = np.empty((1,256), np.uint8)
#     for i in range(256):
#         lookUpTable[0, i] = np.clip(pow(i/255.0, gamma) * 255.0, 0, 255)
#     res = cv2.LUT(new_image, lookUpTable)
#     return res

def resize_image(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    image = cv2.resize(image, (width, height), interpolation = cv2.INTER_NEAREST)
    return image
def remove_shadows(img):
    dilated = cv2.dilate(img, np.ones((9,9), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(img, bg)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return norm

def get_countours(img):
    area_th_max = 120*120
    area_th_min = 50*50
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    contours = [c for n, c in enumerate(contours) if area_th_min < areas[n] < area_th_max]
    return contours



def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    scale = 0.2
    scaled = resize_image(image, scale)
    cv2.imshow("window", scaled)


    #TODO: Implement detection method.
    #remove background:
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 200, 1, cv2.THRESH_BINARY_INV)
    # black = np.zeros(img.shape).astype(img.dtype)
    # result = cv2.bitwise_or(img, black, mask = thresh)

    #another try:


    #convert to grey and threshold:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    without_shadows = remove_shadows(gray)
    img_background = cv2.medianBlur(without_shadows, 5)
    # img_background = cv2.bilateralFilter(img_background, 9, 20, 30)
    img_background = cv2.fastNlMeansDenoising(img_background, None, 5, 7, 21)
    canny = cv2.Canny(img_background, 50, 100, 10)
    # # adaptiveThresh = cv2.adaptiveThreshold(img_background, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    img_background = cv2.dilate(canny, np.ones((2,2)), iterations=8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(img_background, cv2.MORPH_CLOSE, kernel)
    #
    #
    #
    result = cv2.bitwise_and(image, image, mask=morphed)





    # cv2.imshow("result1", resize_image(gray, scale))
    # cv2.imshow("result2", resize_image(img_background, scale))
    # cv2.imshow("result3", resize_image(morphed, scale))
    # cv2.imshow("result4", resize_image(result, scale))
    cv2.imwrite('00_masked.jpg', result)
    cv2.waitKey(0)
    
    red = 0
    yellow = 0
    green = 0
    purple = 0

    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    detect("data/07.jpg")

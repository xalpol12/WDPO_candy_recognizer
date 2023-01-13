import json
from pathlib import Path
from typing import Dict

import click
import cv2
import numpy as np
from tqdm import tqdm


hsv_config_low = np.array([[114, 128, 78], [87, 220, 132],
                           [0, 134, 43], [123, 77, 38]])
hsv_config_high = np.array([[125, 239, 246], [115, 255, 190],
                            [95, 255, 93], [180, 255, 117]])
areas = np.array([[0.0001], [0.00004],
                  [0.00006], [0.00005]])

def process_image(image, low_HSV, high_HSV):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    filtered = cv2.bilateralFilter(gray, 5, 20, 10)
    threshold = cv2.inRange(filtered, low_HSV, high_HSV)
    return threshold

def create_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel)
    return mask_open

def get_contours(mask_open):
    contours, _ = cv2.findContours(mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def get_candy_count(contours, image, area):
    height = 0;
    width = 0
    height, width = image.shape[:2]
    candy_count = 0
    for contour in contours:
        if cv2.contourArea(contour) > area * height * width:
            candy_count += 1
    return candy_count


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

    #TODO: Implement detection method.
    candies = [0, 0, 0, 0]
    image = cv2.imread(img_path)
    for i in range(len(hsv_config_low)):
        in_range = process_image(image, hsv_config_low[i], hsv_config_high[i])
        mask = create_mask(in_range)
        contours = get_contours(mask)
        count = get_candy_count(contours, image, areas[i])
        candies[i] = count
    return {'red': candies[0], 'yellow': candies[1], 'green': candies[2], 'purple': candies[3]}


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
    main()

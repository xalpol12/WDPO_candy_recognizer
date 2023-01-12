import json
from pathlib import Path
from typing import Dict

import click
import cv2
import numpy as np
from tqdm import tqdm

#HSV values for different colors:
#green low_hsv = [44, 121, 35]  high_hsv = [86, 255, 229]
#purple (do sprawdzenia) low_hsv = [106, 52, 49] high_hsv = [178, 174, 115]
#red low_hsv = [106, 155, 62] high_hsv = [123, 255, 252]
#yellow low_hsv = [88, 163, 88] high_hsv = [115, 239, 246]

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

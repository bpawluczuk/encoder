
import cv2
from pathlib import Path

from lib.util_face import getFaceAndCoordinates
from lib.utils import get_image_paths


def convert_one_image(source_image):

    assert source_image.shape == (512, 512, 3)

    resultFace = getFaceAndCoordinates(source_image)

    result = None

    if resultFace is not None:
        xmin, ymin, xmax, ymax, h, w, face = resultFace

        source_image_face = cv2.resize(face, (int(128), int(128)))
        result = source_image_face

    return result


source_dir = Path('dataset/frames/laura_512')
output_dir = Path('dataset/frames/laura_face')


for fn in get_image_paths(source_dir):

    image = cv2.imread(fn)
    new_image = convert_one_image(image)

    if new_image is not None:
        output_file = output_dir / Path(fn).name
        cv2.imwrite(str(output_file), new_image)
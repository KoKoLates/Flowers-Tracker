import numpy as np
from PIL import Image

class Config:
    """ Some config about class and anchor """
    CLASSES = {0: 'flower'}
    ANCHORS = np.array(
        [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 
         72, 146, 142, 110, 192, 243, 459, 401],
        dtype=np.float32
    )


def letterBox_image(image, size):
    """ resize image with unchanged aspect ratio using padding """
    image_w, image_h = image.size
    weight, height = size
    new_weight = int(image_w * min(weight * 1.0 / image_w, height * 1.0 / image_h))
    new_height = int(image_h * min(weight * 1.0 / image_w, height * 1.0 / image_h))
    resized_image = image.resize((new_weight, new_height), Image.BICUBIC)

    boxed_image = Image.new('RGB', size=(128, 128, 128))
    boxed_image.paste(resized_image, ((weight - new_weight) // 2, (height - new_height) // 2))
    return boxed_image

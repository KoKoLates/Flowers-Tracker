import cv2
import argparse

def image_argumenrs():
    """
    Give the arguments parameter of input image path
    and label text path
    """
    parse = argparse.ArgumentParser()
    parse.add_argument('--image_path', type=str, default='./tools/images/test.jpg')
    parse.add_argument('--label_path', type=str, default='./tools/images/test.txt')
    return parse.parse_args()

if __name__ == '__main__':
    """
    Drawing the labeling rectangle boxes.
    """
    arguments = image_argumenrs()
    image = cv2.imread(arguments.image_path)
    image_width, image_heigth = image.shape[1], image.shape[0]
    resize_scale = 0.5
    with open(arguments.label_path) as f:
        for line in f:
            data = (line.strip()).split()
            bbox_width = float(data[3]) * image_width
            bbox_height = float(data[4]) * image_heigth
            center_x = float(data[1]) * image_width
            center_y = float(data[2]) * image_heigth

            min_x, min_y = center_x - (bbox_width / 2), center_y - (bbox_height / 2)
            max_x, max_y = center_x + (bbox_width / 2), center_y + (bbox_height / 2)
            cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), 
                          (0, 255, 255), 2)
            
        f.close()
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

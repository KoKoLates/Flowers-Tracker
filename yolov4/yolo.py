
import os, colorsys, random
import numpy as np, tensorflow as tf

from keras import backend as K
from keras.models import load_model

from yolov4.model import yolo_eval, Mish
from yolov4.utils import letterBox_image



class YOLO(object):
    def __init__(self, model_path:str, score:float=0.5) -> None:
        self.model_path = model_path
        self.score, self.iou = score, 0.5
        self.class_names = self._get_classes('../cfg/classes.txt')
        self.anchors = self._get_anchors('../cfg/anchors.txt')
        self.sess = K._get_session()
        self.model_image_size = (608, 608) # fixed size or (None, None)
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self._generate()

    def _generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.yolo_model = load_model(model_path, custom_objects={'Mish': Mish}, compile=False)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        colors = self._color(len(self.class_names))
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(
            self.yolo_model.output, self.anchors, len(self.class_names), 
            self.input_image_shape, score_threshold=self.score, iou_threshold=self.iou
        )
        return boxes, scores, classes, colors

    def _color(self, num:int):
        hsv_tuples = [(1.0 * x / num, 1., 1.) for x in range(num)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.seed(10101)
        random.shuffle(colors)
        random.seed(None)
        return colors
    

    def _get_classes(self, file_path:str):
        """
        Get the classes name (type) from the indicate file.
        :param file_path (str): the path of indicated file.
        """
        classes_path = os.path.expanduser(file_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    def _get_anchors(self, file_path:str):
        """
        Get the anchors list from the indicate file.
        :param file_path (str): the path of indicated file.
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors
    
    def detect_image(self, image):
        """
        """
        if self.is_fixed_size:
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterBox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image = letterBox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # recognition
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        
        return_boxes, return_scores, return_class_names = [], [], []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if predicted_class != 'person':  # Modify to detect other classes.
                continue
            box = out_boxes[i]
            score = out_scores[i]
            x, y, w, h = int(box[1]), int(box[0]), int(box[3] - box[1]), int(box[2] - box[0])
            if x < 0:
                w = w + x; x = 0
            if y < 0:
                h = h + y; y = 0
            return_boxes.append([x, y, w, h])
            return_scores.append(score)
            return_class_names.append(predicted_class)
        
        return return_boxes, return_scores, return_class_names
        

    def close_session(self) -> None:
        self.sess.close() 

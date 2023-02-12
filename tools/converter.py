#! /usr/bim/env python3
import os, numpy as np
from args import converter_arguments

from keras import backend as K
from keras.layers import Input
from yolov4.model import yolo_eval, yolo4_body

class Yolo4(object):
    def __init__(self, score:float, iou:float, input_size,
                 model_path:str, weights_path:str, 
                 classes_path:str, anchors_path:str, gpu_num:int=1) -> None:
        self.score, self.iou = score, iou
        self.input_size = input_size
        self.model_path = model_path
        self.weights_path = weights_path
        self.classes_path = classes_path
        self.anchors_path = anchors_path
        self.gpu_num = gpu_num
        self.load_yolo()

    def load_yolo(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.class_names = self.get_class(self.classes_path)
        self.anchors = self.get_anchors(self.anchors_path)

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        self.sess = K.get_session()

        # Load model, or construct model and load weights.
        self.yolo4_model = yolo4_body(Input(shape=(self.input_size[0], self.input_size[1], 3)), num_anchors//3, num_classes)

        # Read and convert darknet weight
        self.load_weights(self.yolo4_model, self.weights_path)

        self.yolo4_model.save(self.model_path)

        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(
            self.yolo4_model.output, 
            self.anchors, 
            len(self.class_names), 
            self.input_image_shape, 
            score_threshold=self.score
        )
        print('Done.')

    def load_weights(self, model, weights_file):
        print('[INFO] Loading the weights.')
        wf = open(weights_file, 'rb')
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
        

        j = 0
        for i in range(110):
            conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
            bn_layer_name = 'batch_normalization_%d' % j if j > 0 else 'batch_normalization'

            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if i not in [93, 101, 109]:
                # darknet weights: [beta, gamma, mean, variance]
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                # tf weights: [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_name)
                j += 1
            else:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if i not in [93, 101, 109]:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        assert len(wf.read()) == 0, 'failed to read all data'
        wf.close()

    def get_class(self, file_path:str):
        classes_path = os.path.expanduser(file_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self, file_path:str):
        anchors_path = os.path.expanduser(file_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def close_session(self):
        self.sess.close()

if __name__ == '__main__':
    args = converter_arguments()
    model_path = args.model_path
    weight_path = args.weight_path
    classe_path = args.class_path
    anchor_path = args.anchor_path

    score, iou = 0.5, 0.5
    model_image_size = (608, 608) # To be re-assign

    yolov4_model = Yolo4(score, iou, model_image_size ,
                         model_path, weight_path, classe_path, anchor_path)
    yolov4_model.close_session()
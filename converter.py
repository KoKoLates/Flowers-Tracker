
import tensorflow as tf,yolo.utils as utils

from arguments import converter_arguments
from yolo.yolov4 import YOLO, decode, filter_boxes

def main(args) -> None:
    """
    :param configs (list): the list of configurations of parameter, model and loader file path
    """
    strides, anchors, num_class, scales =  utils.load_config(args)

    input_layer = tf.keras.layers.Input([args.input_size, args.input_size, 3])
    feature_map = YOLO(input_layer, num_class, args.is_tiny)
    bbox_tensors, prob_tensors = [], []

    if args.is_tiny:
        for i, fm in enumerate(feature_map):
            output_tensors = decode(fm, args.input_size // (16 if i == 0 else 32), 
                                    num_class, strides, anchors, i, scales, args.framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    else:
        for i, fm in enumerate(feature_map):
            if i in (0, 1):
                num = 8 if not i else 16
            else:
                num = 32
            output_tensors = decode(fm, args.input_size // num, 
                                    num_class, strides, anchors, i, scales, args.framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])

    pred_bbox, pred_prob = tf.concat(bbox_tensors, axis=1), tf.concat(prob_tensors, axis=1)
    if args.framework == 'tflite':
        pred = (pred_bbox, pred_prob)
    else:
        boxes, pred_conf = filter_boxes(
            pred_bbox, pred_prob, score_threshold=args.score_threshold, 
            input_shape=tf.constant(args.input_size, args.input_size)
        )
        pred = tf.concat([boxes, pred_conf], axis=-1)
    
    model = tf.keras.Model(input_layer, pred)
    utils.load_weight(model, args.weight_path, args.is_tiny)
    model.summary()
    model.save(args.output_path)


if __name__ == '__main__':
    arguments = converter_arguments()
    main(arguments)
    
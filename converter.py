""" Convert the darknet weights into tensorflow model """
import argparse, tensorflow as tf,yolo.utils as utils

from yolo.yolov4 import YOLO, decode, filter_boxes
from ruamel.yaml import YAML

def convert_argument():
    parse = argparse.ArgumentParser()
    parse.add_argument('--tiny', type=bool, default=False, help='yolov4 or tiny')
    return parse.parse_args()

def main(configs: list):
    """
    :param configs (list): the list of configurations of parameter, model and loader file path
    """
    param_config, model_config, loader_config = configs
    args = convert_argument()
    param_config['tiny'] = True if args.tiny else False
    strides, anchors, num_class, scales =  utils.load_config(param_config)


    input_layer = tf.keras.layers.Input([model_config['input_size'], model_config['input_size'], 3])
    feature_map = YOLO(input_layer, num_class, param_config['tiny'])
    bbox_tensors, prob_tensors = [], []

    if param_config['tiny']:
        for i, fm in enumerate(feature_map):
            output_tensors = decode(fm, model_config['input_size'] // (16 if i == 0 else 32), 
                                    num_class, strides, anchors, i, scales, model_config['framework'])
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    else:
        for i, fm in enumerate(feature_map):
            if i in (0, 1):
                num = 8 if not i else 16
            else:
                num = 32
            output_tensors = decode(fm, model_config['input_size'] // num, 
                                    num_class, strides, anchors, i, scales, model_config['framework'])
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])

    pred_bbox, pred_prob = tf.concat(bbox_tensors, axis=1), tf.concat(prob_tensors, axis=1)
    if model_config['framework'] == 'tflite':
        pred = (pred_bbox, pred_prob)
    else:
        boxes, pred_conf = filter_boxes(
            pred_bbox, pred_prob, score_threshold=model_config['score_threshold'], 
            input_shape=tf.constant([model_config['input_size'], model_config['input_size']])
        )
        pred = tf.concat([boxes, pred_conf], axis=-1)
    
    model = tf.keras.Model(input_layer, pred)
    utils.load_weight(model, loader_config['weight_path'], param_config['tiny'])
    model.summary()
    model.save(loader_config['output_path'])


if __name__ == '__main__':
    yaml = YAML(typ='safe')
    yaml.default_flow_style = False
    with open('./cfg/convert_config.yaml', 'r') as config_file:
        config, *model_configs = list(yaml.load_all(config_file))

    main(model_configs)
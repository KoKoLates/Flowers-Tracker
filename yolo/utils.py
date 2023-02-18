
import numpy as np

from yolo.config import cfg

def load_config(config: dict):
    """ Loading the model configuration """
    strides = np.array(cfg.YOLO.STRIDES_TINY if config['tiny'] else cfg.YOLO.STRIDES)
    anchors = load_anchors(
        cfg.YOLO.ANCHORS_TINY if config['tiny'] else cfg.YOLO.ANCHORS, 
        config['tiny']
    )
    scales = cfg.YOLO.XYSCALE_TINY if config['tiny'] else cfg.YOLO.XYSCALE
    num_class = len(config['class_file'])
    return strides, anchors, num_class, scales


def load_anchors(anchors_cfg, tiny:bool=False):
    anchors = np.array(anchors_cfg)
    return anchors.reshape(2, 3, 2) if tiny else anchors.reshape(3, 3, 2)


def read_class_name(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for id, name in enumerate(data):
            names[id] = name.strip('\n')
    return names

def load_weight(model, weights_file:str, is_tiny:bool=False) -> None:
    """"""
    layer_size = 13 if is_tiny else 110
    output_pos = [9, 12] if is_tiny else [63, 101, 109]
    file = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(file=file, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        conv_layer_name = f'conv2d_{i}' if i > 0 else 'conv2d'
        bn_layer_name = f'batch_normalization_{j}' if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            bn_weights = np.fromfile(file=file, dtype=np.float32, count=4 * filters)
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(file=file, dtype=np.float32, count=filters)

        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(file, dtype=np.float32, count=np.product(conv_shape))
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])
    
    file.close()

import numpy as np

from yolo.config import cfg

def load(config: dict, type:str):
    if type not in ['convert', 'tracker']:
        raise ValueError
    


def load_config(args):
    """ Loading the model configuration """
    strides = np.array(cfg.YOLO.STRIDES_TINY if args.is_tiny else cfg.YOLO.STRIDES)
    anchors = load_anchors(
        cfg.YOLO.ANCHORS_TINY if args.is_tiny else cfg.YOLO.ANCHORS, 
        args.is_tiny
    )
    scales = cfg.YOLO.XYSCALE_TINY if args.is_tiny else cfg.YOLO.XYSCALE
    num_class = len(read_class_names(args.class_names))
    return strides, anchors, num_class, scales


def load_anchors(anchors_cfg, tiny:bool=False):
    anchors = np.array(anchors_cfg)
    return anchors.reshape(2, 3, 2) if tiny else anchors.reshape(3, 3, 2)


def read_class_names(class_file_name):
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

def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        width = xmax - xmin
        height = ymax - ymin
        box[0], box[1], box[2], box[3] = xmin, ymin, width, height

    return bboxes

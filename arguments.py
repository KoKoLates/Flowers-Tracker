import argparse

def converter_arguments():
    parse = argparse.ArgumentParser()
    parse.add_argument('--weight_path', type=str, default='./cfg/yolov4_final.weights')
    parse.add_argument('--output_path', type=str, default='./checkpoints/yolov4')
    parse.add_argument('--class_names', type=str, default='./cfg/object.names')
    parse.add_argument('--framework', type=str, default='tf', help='tf, trt, tflite')
    parse.add_argument('--input_size', type=int, default=416)
    parse.add_argument('--is_tiny', type=bool, default=False)
    parse.add_argument('--score_threshold', type=float, default=0.3)
    return parse.parse_args()


def tracker_arguments():
    parse = argparse.ArgumentParser()
    parse.add_argument('--video', type=str, default='./video/test.mp4')
    parse.add_argument('--output', type=bool, default=False)
    parse.add_argument('--output_format', type=str, default='XVID', help='XVID for *.avi format')
    parse.add_argument('--output_path', type=str, default='./video/outputs/test.avi')
    parse.add_argument('--weight_path', type=str, default='./checkpoints/yolov4')
    parse.add_argument('--class_names', type=str, default='./cfg/object.names')
    parse.add_argument('--framework', type=str, default='tf', help='tf, trt, tflite')
    parse.add_argument('--input_size', type=int, default=416)
    parse.add_argument('--iou', type=float, default=0.45)
    parse.add_argument('--score', type=float, default=0.5)
    parse.add_argument('--is_tiny', type=bool, default=False)
    parse.add_argument('--info_shown', type=bool, default=False)
    parse.add_argument('--count_shown', type=bool, default=True)
    parse.add_argument('--video_shown', type=bool, default=True)
    return parse.parse_args()
    
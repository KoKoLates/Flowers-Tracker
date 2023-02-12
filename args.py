import argparse


def tracker_arguments():
    """
    Setup all needed arguments configuration for tracker
    """
    parse = argparse.ArgumentParser()
    parse.add_argument('--video', type=str, default='test.mp4', help='input data video *.mp4')
    parse.add_argument('--mini_score', type=float, default=0.3, help='lowest tracking score')
    parse.add_argument('--model_file', type=str, default='./cfg/model.h5', help='object detection model file')
    parse.add_argument('--faeture', type=str, default='./cfg/market1501.pb', help='tracking target model file')
    return parse.parse_args()

def converter_arguments():
    """
    Setup all needed arguments configuration for converter
    """
    parse = argparse.ArgumentParser()
    parse.add_argument('--model_path', type=str, default='./cfg/model.h5 ', help='model file path')
    parse.add_argument('--weight_path', type=str, default='./cfg/yolov4.weight', help='weight file from YOLO')
    parse.add_argument('--class_name', type=str, default='./cfg/classes.txt', help='classes name file path')
    parse.add_argument('--anchor_name', type=str, default='./cfg/anchors.txt', help='anchor name file path')
    return parse.parse_args()
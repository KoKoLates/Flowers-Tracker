# YOLOv4 Training with Darknet
In this repository, the [`darknet`](https://github.com/AlexeyAB/darknet) model is used for the YOLOv4 and YOLOv4 tiny. Here are some instruction for custom YOLOv4 tiny setups and training steps.

## Environment Setup
* `$ git clone https://github.com/AlexeyAB/darknet`
* Modifying the Makefile in the darknet master for setup `GPU`, `CUDNN`, `CUDNN_HALF` and `OpenCV`.
```shell
sed -i "s/GPU=0/GPU=1/g" darknet/Makefile
sed -i "s/CUDNN=0/CUDNN=1/g" darknet/Makefile
sed -i "s/CUDNN_HALF=0/CUDNN_HALF=1/g" darknet/Makefile
sed -i "s/OPENCV=0/OPENCV=1/g" darknet/Makefile
```
* `$ cd darknet && make`

## Dataset Configuration
* Create a custom object detection directory for storing the dataset (images and labels), weights and miscellenous configurations.
* `$ cd .. && mkdir object_detection && cd object_detection`
* Create weights and dataset directory.
```python
import os, shutil

if not os.path.exists('object_detection'):
    os.mkdir('object_detection')
if not os.path.exists('object_detection/cfg'):
    os.mkdir('object_detection/cfg') 
    os.mkdir('object_detection/weights')
if not os.path.exists('object_detection/cfg/object.data'):
    shutil.copyfile('darknet/cfg/coco.data', 
                    'object_detection/cfg/object.data')
if not os.path.exists('object_detection/cfg/object.name'):
    shutil.copyfile('darknet/cfg/coco.names', 
                    'object_detection/cfg/object.names')
```
* Create train, validation and test directory. And put the images and label data in the right path for the program to obtain the dataset correctly.
```
$ env:USERPROFILE
├── darknet
└── object_detection 
    ├── cfg
    |   ├── weights/      # weights directory
    |   ├── object.name   # classes name of dtected object
    |   ├── object.data   # path configuration
    |   ├── train.txt     # path of train images  
    |   ├── valid.txt     # path of valid images
    |   ├── yolov4-tiny-obj.cfg
    |   └── yolov4-tiny.conv.29
    ├── train             # train dataset directory
    |   ├── images
    |   └── labels
    ├── valid             # valid dataset directory
    |   ├── images
    |   └── labels
    └── test
```
* Modifying `object.data` files. Notes that the relative path is based on the path of darket training program.
```
classes = 1 
train = ../object_detection/cfg/train.txt
valid = ../object_detection/cfg/valid.txt
names = ../object_detection/cfg/object.name
backup = ../object_detection/cfg/weights/
```
* `$ cp ../darknet/cfg/yolov4-tiny-custom.cfg cfg/yolov4-tiny-obj.cfg`
* Copy the `darknet/cfg/yolov4-tiny-custom.cfg` into `cfg` directory and modify the filters and classes into proper values for object detection training.

## Training Models
```
./darknet detector train ../object_detection/cfg/object.data
                         ../object_detection/cfg/yolov4-tiny-obj.cfg
                         ../object_detection/cfg/yolov4-tiny.conv.29
```

## Testing Models
The weights will be store at `cfg/weights` directory. After modify the `object.data` files into testing mode. Ones could use the image in the test directory for testing the model performaces.
```
./darknet detector test ../object_detection/cfg/object.data
                        ../object_detection/cfg/yolov4-tiny-obj.cfg
                        ../object_detection/cfg/weights/yolov4-tiny-obj_final.weights
                        ../object_detection/test/test_image_1.jpg
```
* Notes that the result will be store at `darknet` directory.
* The `mAP` and `recall` could be calculated as the performaces metric as well/
```bash
# mAP calculate
./darknet detector map ../object_detection/cfg/object.data
                       ../object_detection/cfg/yolov4-tiny-obj.cfg
                       ../object_detection/cfg/weights/yolov4-tiny-obj_final.weights

# recall calculate
./darknet detector recall ../object_detection/cfg/object.data
                          ../object_detection/cfg/yolov4-tiny-obj.cfg
                          ../object_detection/cfg/weights/yolov4-tiny-obj_final.weights
```

## Reference
* [yolov4 tiny darknet](https://github.com/AlexeyAB/darknet)

# Flowers Tracker
The flower tracker and counter based on the mulitple object tracking that implemented with `YOLOv4`, `Deep SORT` and `TensorFlow`. 

![image](./assets/flowers.png)
![image](./assets/tracking.gif)

## Getting Started
Install the proper dependencies via pip or anaconda. Notes that tensorflow 2 packages required a pip version larger than 19.0.
```shell
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

## Running the Tracker with YOLOv4 tiny
The tracker allow user to use detection like YOLOv3, YOLOv4 and tiny. The following command is used for YOLOv4 tiny model. `Yolov4-tiny` allows you to obtain a higher speed (FPS) for the tracker at a slight cost to accuracy. Make sure that you have downloaded the tiny weights file and added it to the `cfg` folder in order for commands to work.
```shell
# convert yolov4-tiny model
python converter.py --weight_path ./data/yolov4-tiny.weights --output_path ./checkpoints/yolov4-tiny --model yolov4 --tiny

# Run yolov4-tiny object tracker
python tracker.py --weight_path ./checkpoints/yolov4-tiny --model yolov4 --video ./data/video/test.mp4 --output_path ./outputs/tiny.avi --tiny
```

## Citation

__YOLOv4__
```
@article{bochkovskiy2020yolov4,
  title={Yolov4: Optimal speed and accuracy of object detection},
  author={Bochkovskiy, Alexey and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2004.10934},
  year={2020}
}
```

__Deep SORT__
```
@inproceedings{wojke2017simple,
  title={Simple online and realtime tracking with a deep association metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE international conference on image processing (ICIP)},
  pages={3645--3649},
  year={2017},
  organization={IEEE}
}

@inproceedings{wojke2018deep,
  title={Deep cosine metric learning for person re-identification},
  author={Wojke, Nicolai and Bewley, Alex},
  booktitle={2018 IEEE winter conference on applications of computer vision (WACV)},
  pages={748--756},
  year={2018},
  organization={IEEE}
}
```

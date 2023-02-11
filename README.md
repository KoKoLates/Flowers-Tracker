# flowers tracker
A tracker and counter for Eustoma Grandiflorum flowers by using `YOLOv4` and `Deep SORT`.

## Requirement
* OpenCV
* sklearn
* numpy 1.15.0
* torch
* tensorflow

## Quick Start
__0. Requirements Setup__
```
pip install -r requirements.txt
```

## Citation

__YOLOv4__

    @misc{bochkovskiy2020yolov4,
    title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
    author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
    year={2020},
    eprint={2004.10934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
    }

__Deep_SORT__

    @inproceedings{Wojke2017simple,
    title={Simple Online and Realtime Tracking with a Deep Association Metric},
    author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
    booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
    year={2017},
    pages={3645--3649},
    organization={IEEE},
    doi={10.1109/ICIP.2017.8296962}
    }

    @inproceedings{Wojke2018deep,
    title={Deep Cosine Metric Learning for Person Re-identification},
    author={Wojke, Nicolai and Bewley, Alex},
    booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
    year={2018},
    pages={748--756},
    organization={IEEE},
    doi={10.1109/WACV.2018.00087}
    }

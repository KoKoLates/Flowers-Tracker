import os, tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import cv2, time
import numpy as np, matplotlib.pyplot as plt, yolo.utils as utils

from yolo.yolov4 import filter_boxes
from PIL import Image

from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
from tensorflow.python.saved_model import tag_constants

from deep_sort import preprocessing, nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection

from tools import generate_detections as gdet
from arguments import tracker_arguments


def main(args) -> None:
    """"""
    # Definination of the parameters
    max_cos_distance:float = 0.4
    nn_budget, nms_max_overlap:float = None, 1.0

    # Initialize the deep sort
    encoder = gdet.create_box_encoder('./cfg/mars-small128.pb', batch_size=1)

    # Calculating the cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cos_distance, nn_budget)
    # Initialize the tracker
    tracker = Tracker(metric=metric)

    # loading the configuration for object detector
    _config = ConfigProto()
    _config.gpu_options.allow_growth = True
    session = InteractiveSession(config=_config)
    strides, anchors, num_class, scales =  utils.load_config(args)
    input_size, video_path = args.input_size, args.video

    # laoding the tflite model if the flag is set
    if args.framework in ['tflite']:
        interpreter = tf.lite.Interpreter(model_path=args.weight_path)
        interpreter.allocate_tensors()
        print(f'[INFO] INPUT_DETAILS -- {interpreter.get_input_details()}.')
        print(f'[INFO] OUTPUT_DETAILS -- {interpreter.get_output_details()}.')
    else:
        saved_model_loaded = tf.save_model.load(args.weight_path, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # Setup the video capture and writer 
    try:
        video = cv2.VideoCapture(int(video_path))
    except:
        video = cv2.VideoCapture(video_path)

    if args.output:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*args.output_format)
    video_writer = cv2.VideoWriter(
        args.output_path, codec, fps, (width, height)
    ) if args.output else None

    # Running the video
    frame_num = 0
    while True:
        ret, frame = video.read()
        if not ret:
            print('Video has ended or failed.')
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2RGB)
        image = Image.fromarray(frame)
        frame_num += 1
        print(f'Frames -- # {frame_num}')

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size)) / 255.0
        image_data = image_data[np.newasis, ...].astype(np.float32)
        start_time = time.time()

        # Run detection 
        if args.framework in ['tflite']:
            interpreter.set_tensor(interpreter.input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(interpreter.output_details[i]['index']) 
                    for i in range(len(interpreter.output_details))]
            boxes, pred_conf = filter_boxes(
                pred[0], pred[1], score_threshold=0.25,
                input_shape=tf.constant([input_size, input_size])
            )
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
            ),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=args.iou,
            score_threshold=args.score
        )

        # Convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(args.class_names)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
         # allowed_classes = ['person'] # indicate specific class

        names, delete_indx = [], []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                delete_indx.append(i)
            else:
                names.append(class_name)

        names = np.array(names)
        count = len(names)
        if args.count_shown:
            cv2.putText(
                frame, f'Being Tracked -- {count}', (5, 35), 
                cv2.FONT_HERSHEY_PLAIN, 1, (46, 41, 43), 1.5
            )
            print(f'{count} objects being tracked.')

        bboxes = np.delete(bboxes, delete_indx, axis=0)
        scores = np.delete(scores, delete_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) 
                      for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)), 
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            cv2.putText(
                frame, f'{class_name} - {str(track.track_id)}'
                ,(int(bbox[0]), int(bbox[1] - 10)), 0, 0.5, (255, 255, 255), 2
            )

            if args.info_shown:
                print(f'Track ID -- {str(track.track_id)} | Class -- {class_name}' + \
                      f'| BBox -- ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}).')
                
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print(f'FPS -- {fps:.2f}.')
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if args.video_shown: cv2.imshow('Output Video', result)
        if args.out: video_writer.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    video.release()
    video_writer.release()
    cv2.destroyAllWindows()
        

if __name__ == '__main__':
    arguments = tracker_arguments()
    main(arguments)


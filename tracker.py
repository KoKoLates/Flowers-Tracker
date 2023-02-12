#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2, time, numpy as np

from PIL import Image
from yolov4.yolo import YOLO
from args import tracker_arguments
from tools import generate_detections

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection

if __name__ == '__main__':
    # Arguments parameter configuration
    args = tracker_arguments()

    # Deep SORT and Tracker
    encoder = generate_detections.create_box_encoder(args.feature, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric('cosine', args.mini_score, None)
    tracker = Tracker(metric)

    # loding the model
    yolo = YOLO(args.model_file, args.mini_score)

    # Reading the video
    video = cv2.VideoCapture(args.video)

    # Saving the video
    fourcc = cv2.VideoWriter_fourcc(*'X264') # MPEG-4 for *.mp4 / Using *XVID for MPEG-4 *.avi
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (
        int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), 
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    video_out = cv2.VideoWriter('output.mp4', fourcc, fps, size)

    # Open the video and tracking for each fram
    while True:
        # Read in frame
        ret, frame = video.read()
        if not ret:
            break

        prev_time = time.time()

        # Image Transformation
        image = Image.fromarray(frame[...,::-1]) # BGR to RGB
        boxes, scores, classes, colors = yolo.detect_image(image)

        # Feature Extraction and Detection
        features = encoder(frame, boxes)
        detections = []
        for bbox, score, classe, color, feature in zip(boxes, scores, classes, colors, features):
            detections.append(Detection(bbox, score, classes, color, feature))

        # maximum suppression
        boxes = np.array([detect.tlwh for detect in detections])
        scores = np.array([detect.score for detect in detections])
        indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        tracker.predict()
        tracker.update(detections)

        # printing messages
        track_count, track_total = 0, 0
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1: continue
            y1, x1, y2, x2 = np.array(track.to_tlbr(), dtype=np.int32)
            cv2.putText(
                frame, f'No. {track.track_id}', (y1, x1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, (255, 255, 255), 1, lineType=cv2.LINE_AA
            )
            if track.track_id > track_total: track_total = track.track_id
            track_count += 1

        # printeing detection messages
        total_count = {}
        for detect in detections:
            y1, x1, y2, x2 = np.array(detect.to_tlbr(), dtype=np.int32)
            caption = '{} {:.2f}'.format(detect.classe, detect.score) if detect.classe else detect.score
            cv2.rectangle(frame, (y1, x1), (y2, x2), detect.color, 2)
            text_size = cv2.getTextSize(caption, 0, 0.4, thickness=2)[0]
            cv2.rectangle(frame, (y1, x1), (y1 + text_size[0], x1 + text_size[1] + 8), detect.color, -1)
            cv2.putText(
                frame, caption, (y1, x1 + text_size[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 2, lineType=cv2.LINE_AA
            )
            if detect.classe not in total_count: total_count[detect.classe] = 0
            total_count[detect.classe] += 1

        trackTotalStr = 'Track Total: %s' % str(track_total)
        cv2.putText(frame, trackTotalStr, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 0, 255), 1, cv2.LINE_AA)

        trackCountStr = 'Track Count: %s' % str(track_count)
        cv2.putText(frame, trackCountStr, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 0, 255), 1, cv2.LINE_AA)

        totalStr = ""
        for k in total_count.keys(): totalStr += '%s: %d    ' % (k, total_count[k])
        cv2.putText(frame, totalStr, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 0, 255), 1, cv2.LINE_AA)

        curr_time = time.time()
        exec_time = curr_time - prev_time
        print("Cost Times: %.2f ms" %(1000*exec_time))

        video_out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    video_out.release()
    cv2.destroyAllWindows()

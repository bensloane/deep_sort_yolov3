#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys

import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import argparse
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

def main(yolo):
    '''
    Command line options
    '''
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model", type="str", required=True,
        help="path to model weight file, default " + yolo.model_path
    )

    ap.add_argument(
        "--anchors", type="str", required=True,
       help="path to anchor definitions, default " + yolo.anchors_path
    )

    ap.add_argument(
        "--detection_threshold", type=float, required=False, default=0.3, 
        help="Confidence threshold for detection.
    )

    ap.add_argument(
        "--iou_threshold", type=float, required=False, default=0.45, 
        help="IOU threshold for detection.
    )   

    ap.add_argument(
        "--input", type="str", required=False, default=None, 
        help="Path to a video file to use as input inplace of a webcam. If None then use webcam."
    )

    ap.add_argument(
        "--output", type="str", required=False, default=None, 
        help="Name of the output video file. If None then no video is written."
    )

    ap.add_argument(
        "--fps", type=int, required=False, default=15, 
        help="Specifies the frames per second to use for writing to output video. By default uses 15 \
        You can find the fps of the video capture using cv2.VideoCapture(...).get(CAP_PROP_FPS)"
    )

    # TODO: Optionally control how often detection is run 
    # ap.add_argument(
    #     "--skip_frames", type=int, required=False, default=None, 
    #     help="Specifies the number of frames to skip until running detection. The higher the number \
    #     the more frames that will be skipped and thus less time spent running inference."
    # )

    args = vars(ap.pars_args())

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
 
    # Set the yolo attributes
    if args["model"]:
        yolo.model = args["model"]

    if args["anchors"]:
        yolo.anchors_path = args["anchors"]

    if args["detection_threshold"]:
        yolo.score = args["detection_threshold"]:

    if args["iou_threshold"]:
        yolo.iou = args["iou_threshold"]:

    # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    
    # Save output to video file
    writeVideo_flag = False
    if args["output"]:
        writeVideo_flag = True

    # Use a file instead of webcam as input
    if args["input"]:
        video_capture = cv2.VideoCapture(args["input"])
    else:
        video_capture = cv2.VideoCapture(0)

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(args["output"], fourcc, args["fps"], (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
        
    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break;
        t1 = time.time()

        image = Image.fromarray(frame

        boxs = yolo.detect_image(image)
       # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        
        if not args["input"]:
            cv2.imshow('', frame)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if not args["input"] & cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())

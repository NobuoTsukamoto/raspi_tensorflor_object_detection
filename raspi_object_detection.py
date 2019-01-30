#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2019 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import collections
import random
import time

import cv2

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image
from collections import defaultdict
from datetime import datetime as dt

# This is needed since the notebook is stored in the object_detection folder.
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

sys.path.append("/home/pi/models/research/object_detection")
from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = '/home/pi/models/research/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/pi/models/research/object_detection/data', 'mscoco_label_map.pbtxt')

def load_image_into_numpy_array2(image):
    return np.asarray(image).astype(np.uint8)

def main():
    WINDOW_NAME = 'Tensorflow object detection'
    freq = cv2.getTickFrequency()

    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 100, 200)

    image = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(image, 'Loadg ...', (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow(WINDOW_NAME, image)
    for i in range(20):
        cv2.waitKey(10)

    # Load a (frozen) Tensorflow model into memory.
    print('Load a (frozen) Tensorflow model into memory.')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Loading label map
    print('Loading label map.')
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


    with detection_graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            print('Get handles to input and output tensors.')
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Start VideoCapture.
            print('Start VideoCapture.')
            cap = cv2.VideoCapture(0)
            cap.set(3, 640)
            cap.set(4, 480)

            # Run inference
            while (True):
                # Capture frame-by-frame
                ret, frame = cap.read()

                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                # image_np = load_image_into_numpy_array2(image)

                # bgr -> rgb
                image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)

                start_time = cv2.getTickCount()

                # inference
                print('sess.run in.')
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image_np, 0)})
                print('sess.run out.')

                end_time = cv2.getTickCount()

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                        frame,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index,
                        instance_masks=output_dict.get('detection_masks'),
                        use_normalized_coordinates=True,
                        line_thickness=4)

                # Draw FPS
                frame_rate = 1 / ((end_time - start_time) / freq)
                cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate),
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

                # Display the resulting frame
                cv2.imshow(WINDOW_NAME, frame)
#                if cv2.waitKey(10) & 0xFF == ord('q') or video_getter.stopped:
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                for i in range(10):
                    ret, frame = cap.read()

    # When everything done, release the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



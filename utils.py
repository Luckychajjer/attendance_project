# *******************************************************************
#
# Author : Thanh Nguyen, 2018
# Email  : sthanhng@gmail.com
# Github : https://github.com/sthanhng
#
# BAP, AI Team
# Face detection using the YOLOv3 algorithm
#
# Description : utils.py
# This file contains the code of the parameters and help functions
#
# *******************************************************************


import datetime
import numpy as np
import cv2,os
import imagehash
import pandas as pd
from PIL import Image

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

# Default colors
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)


# -------------------------------------------------------------------
# Help functions
# -------------------------------------------------------------------

# Get the names of the output layers
def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    #return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return [(layers_names[i - 1], print(i))[0] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def draw_predict(frame, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)

    text = '{:.2f}'.format(conf)

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    top = max(top, label_size[1])
    cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                COLOR_WHITE, 1)


def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)
    counter=1
    for i in indices:
        #i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        left, top, right, bottom = refined_box(left, top, width, height)
        # draw_predict(frame, confidences[i], left, top, left + width,
        #              top + height)
        crop_image(frame,left,top,right,bottom,counter)
        counter+=1
        draw_predict(frame, confidences[i], left, top, right, bottom)
    return final_boxes


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._num_frames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._num_frames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._num_frames / self.elapsed()

def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

    right = right + margin

    return left, top, right, bottom

def crop_image(frame,left,top,right,bottom,counter):
    output_folder = "cropped_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List of image files you want to process

    # Crop the region from the original image
    cropped_image = frame[top:bottom, left:right]

    # Save the cropped image in the "cropped_images" folder
    output_path = os.path.join(output_folder, f"cropped_image{counter}.jpg")
    cv2.imwrite(output_path, cropped_image)


# Function to compute the hash of an image
def compute_image_hash(image_path):
    try:
        with Image.open(image_path) as img:
            hash_value = imagehash.average_hash(img)
            return hash_value
    except Exception as e:
        print(f"Error computing hash for {image_path}: {e}")
        return None

# Function to find duplicate images in two folders
def find_duplicate_images(folder1, folder2):
    # Get a list of image files in both folders
    files1 = [f for f in os.listdir(folder1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    files2 = [f for f in os.listdir(folder2) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Compute hashes for images in the first folder
    hashes1 = {compute_image_hash(os.path.join(folder1, f)): f for f in files1}

    # Compare hashes of images in the second folder with the first folder
    duplicate_images = []
    for file2 in files2:
        hash2 = compute_image_hash(os.path.join(folder2, file2))
        if hash2 is not None:
            if hash2 in hashes1:
                duplicate_images.append((file2, hashes1[hash2]))

    return duplicate_images

# Function to update CSV file
def update_csv(csv_file, duplicates):
    if not duplicates:
        return

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Iterate through the duplicate images and update the "ATTENDENCE" column
    for img1, img2 in duplicates:
        # Assuming the CSV file has a "NAME" column that matches image filenames
        img2_filename = os.path.splitext(os.path.basename(img2))[0]
        df.loc[df['NAME'] == img2_filename, 'ATTENDENCE'] = 'PRESENT'

    # Save the updated DataFrame back to the CSV file
    df.to_csv(csv_file, index=False)
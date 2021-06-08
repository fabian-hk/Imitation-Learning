from typing import Tuple
import h5py
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

print(tf.config.list_physical_devices('GPU'))

data_set_file = "data/track_data_2.h5"


def label_to_upper_point(degree: float) -> Tuple[int, int]:
    hyp = 20
    op = np.sin(np.deg2rad(degree)) * hyp
    adj = np.sqrt(np.square(hyp) - np.square(op))
    return round(30 + op), round((55 - hyp) + (hyp - adj))


with h5py.File(data_set_file, "r") as f:
    print(f.keys())

    for key in f.keys():
        print(f[key])

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (64, 60))
    for image, label in zip(f['images'], f['angles']):
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.line(img, label_to_upper_point(label), (30, 55), (0, 255, 0), 2)
        video.write(img)

    video.release()
    print("Video saved")

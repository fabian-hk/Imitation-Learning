from typing import Tuple
import cv2
import numpy as np


def label_to_upper_point(degree: float) -> Tuple[int, int]:
    hyp = 20
    op = float(np.sin(np.deg2rad(degree)) * hyp)
    adj = float(np.sqrt(np.square(hyp) - np.square(op)))
    return round(30 + op), round((55 - hyp) + (hyp - adj))


def draw_angle_in_video(images, y_pred, y_test):
    video = cv2.VideoWriter('visualization.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (64, 60))
    for image, y_, y in zip(images, y_pred, y_test):
        image = image * 255.0
        image = np.asarray(image, dtype=np.uint8)
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.line(img, label_to_upper_point(y), (30, 55), (0, 255, 0), 2)
        cv2.line(img, label_to_upper_point(y_), (30, 55), (0, 0, 255), 2)
        video.write(img)

    video.release()

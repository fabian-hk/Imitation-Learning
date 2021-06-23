from typing import Tuple, List
import cv2
import numpy as np
from PIL import Image

from data_set import DataSet


def label_to_upper_point(degree: float, center: Tuple[int, int]) -> Tuple[int, int]:
    radius = 100
    x = round(center[0] + (radius * np.sin(np.deg2rad(degree))))
    y = round(center[1] - (radius * np.cos(np.deg2rad(degree))))
    return x, y


def draw_angle_in_image(image: np.ndarray, y_true, y_pred=None):
    image = np.asarray(image, dtype=np.uint8)
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (320, 300), interpolation=cv2.INTER_NEAREST)
    center = (round(img.shape[1] / 2.0), round(img.shape[0] / 2.0))
    circle_point = label_to_upper_point(y_true, center)
    cv2.line(img, circle_point, center, (0, 255, 0), 8)
    if y_pred is not None:
        circle_point = label_to_upper_point(y_pred, center)
        cv2.line(img, circle_point, center, (255, 0, 0), 8)
    return img


def visualize_angle_in_mp4(images: np.ndarray, y_true, y_pred=None):
    video = cv2.VideoWriter('visualization.mp4', cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 24, (64, 60))
    if y_pred is not None:
        for image, y, y_ in zip(images, y_true, y_pred):
            img = draw_angle_in_image(image, y, y_)
            video.write(img)
    else:
        for image, y in zip(images, y_true):
            img = draw_angle_in_image(image, y)
            video.write(img)
    video.release()


def visualize_angle_in_gif(images: np.ndarray, y_true, y_pred=None, fn="visualization", duration=40):
    result = []
    if y_pred is not None:
        for image, y, y_ in zip(images, y_true, y_pred):
            img = draw_angle_in_image(image, y, y_)
            img = Image.fromarray(img, mode="RGB")
            result.append(img)
    else:
        for image, y in zip(images, y_true):
            img = draw_angle_in_image(image, y)
            img = Image.fromarray(img, mode="RGB")
            result.append(img)
    result[0].save(fp=f"graphic/{fn}.gif", format="GIF", append_images=result, optimize=False, save_all=True, duration=duration,
                   loop=0)


if __name__ == '__main__':
    df = DataSet(f"data/track_data_2.h5")
    df.preprocessing()

    visualize_angle_in_gif(df.x[:df.train_data_length], df.y[:df.train_data_length])

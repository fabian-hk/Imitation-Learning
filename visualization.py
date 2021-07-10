from typing import Tuple, List
import cv2
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 30})
import numpy as np
from PIL import Image


def label_to_upper_point(degree: float, center: Tuple[int, int]) -> Tuple[int, int]:
    radius = 100
    x = round(center[0] + (radius * np.sin(np.deg2rad(degree))))
    y = round(center[1] - (radius * np.cos(np.deg2rad(degree))))
    return x, y


def draw_angle_in_image(image: np.ndarray, y_true, y_pred=None, resize=True):
    image = np.asarray(image, dtype=np.uint8)
    if len(image.shape) == 2:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.merge((image[0], image[1], image[2]))
    if resize:
        img = cv2.resize(img, (320, 300), interpolation=cv2.INTER_NEAREST)
    center = (round(img.shape[1] / 2.0), round(img.shape[0] / 2.0))
    circle_point = label_to_upper_point(y_true, center)
    cv2.line(img, circle_point, center, (0, 255, 0), 8)
    if y_pred is not None:
        circle_point = label_to_upper_point(y_pred, center)
        cv2.line(img, circle_point, center, (255, 0, 0), 8)
    return img


def visualize_angle_in_mp4(
    images: np.ndarray, y_true, y_pred=None, resize=True, fn="visualization"
):
    shape = (320, 300)
    if not resize:
        shape = (images[0].shape[-1], images[0].shape[-2])
    print(f"Video size: {shape}")
    video = cv2.VideoWriter(
        f"graphic/{fn}.mp4", cv2.VideoWriter_fourcc("M", "P", "4", "V"), 24, shape
    )
    if y_pred is not None:
        for image, y, y_ in zip(images, y_true, y_pred):
            img = draw_angle_in_image(image, y, y_, resize=resize)
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        for image, y in zip(images, y_true):
            img = draw_angle_in_image(image, y, resize=resize)
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video.release()


def visualize_angle_in_gif(
    images: np.ndarray,
    y_true,
    y_pred=None,
    fn="visualization",
    duration=40,
    resize=True,
):
    max_imgs = 3200
    result = []
    if y_pred is not None:
        for image, y, y_ in zip(images[:max_imgs], y_true[:max_imgs], y_pred[:max_imgs]):
            img = draw_angle_in_image(image, y, y_, resize=resize)
            img = Image.fromarray(img, mode="RGB")
            result.append(img)
    else:
        for image, y in zip(images[:max_imgs], y_true[:max_imgs]):
            img = draw_angle_in_image(image, y, resize=resize)
            img = Image.fromarray(img, mode="RGB")
            result.append(img)
    result[0].save(
        fp=f"graphic/{fn}.gif",
        format="GIF",
        append_images=result,
        optimize=False,
        save_all=True,
        duration=duration,
        loop=0,
    )


def data_set_visualization(images: List[np.ndarray], y_true):
    images_preprocessed = []
    for img, y in zip(images, y_true):
        images_preprocessed.append(draw_angle_in_image(img, y, resize=False))

    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    axs[0, 0].imshow(images_preprocessed[0])
    axs[0, 0].set_title(f"Steering Angle {round(y_true[0])}°")
    axs[0, 0].set_xlabel("Pixel x")
    axs[0, 0].set_ylabel("Pixel y")
    axs[0, 1].imshow(images_preprocessed[1])
    axs[0, 1].set_title(f"Steering Angle {round(y_true[1])}°")
    axs[0, 1].set_xlabel("Pixel x")
    axs[0, 1].set_ylabel("Pixel y")
    axs[1, 0].imshow(images_preprocessed[2])
    axs[1, 0].set_title(f"Steering Angle {round(y_true[2])}°")
    axs[1, 0].set_xlabel("Pixel x")
    axs[1, 0].set_ylabel("Pixel y")
    axs[1, 1].imshow(images_preprocessed[3])
    axs[1, 1].set_title(f"Steering Angle {round(y_true[3])}°")
    axs[1, 1].set_xlabel("Pixel x")
    axs[1, 1].set_ylabel("Pixel y")
    plt.show()


def reference_data_set_visualization():
    from data_set import ReferenceDataSet

    df = ReferenceDataSet()

    img = [df.X[0], df.X[300], df.X[50], df.X[1000]]
    y_true = [df.y[0], df.y[300], df.y[50], df.y[1000]]
    data_set_visualization(img, y_true)


if __name__ == "__main__":
    from data_set import CommaAiDataSet

    df = CommaAiDataSet("")
    df.normalize_steering_angle()
    img = [df.X[0], df.X[300], df.X[50], df.X[1000]]
    y_true = [df.y[0], df.y[300], df.y[50], df.y[1000]]
    data_set_visualization(img, y_true)

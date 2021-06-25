from typing import Tuple
import h5py
import numpy as np
from sklearn.utils import shuffle
from PIL import Image
from pathlib import Path

import utils


class DataSet:

    def __init__(self, data_set_file="data/track_data_2.h5"):
        self.x = []
        self.y = []

        with h5py.File(data_set_file, "r") as f:
            for image, label in zip(f['images'], f['angles']):
                self.x.append(image)
                self.y.append(label)

        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.train_data_length = None

    def preprocessing(self, scale=None, output_bins=45):
        if scale is not None:
            # down scale images
            x_tmp = []
            for img in self.x:
                img = Image.fromarray(img)
                img = img.resize(scale, Image.NEAREST)
                x_tmp.append(np.array(img, dtype=np.float32))

            self.x = np.asarray(x_tmp)

        # normalize images
        self.x_normalized = self.x / 255.0

        # split data in train and test set and shuffle them
        split = round(len(self.x_normalized) * 0.7)
        self.train_data_length = split
        x_train = self.x_normalized[:split]
        y_train = self.y[:split]
        self.x_train, self.y_train = shuffle(x_train, y_train, random_state=16)

        self.x_test = self.x_normalized[split:]
        self.y_test = self.y[split:]

        # encode steering wheel angles
        self.y_train = utils.encode_angle(self.y_train, output_bins=output_bins)
        self.y_test = utils.encode_angle(self.y_test, output_bins=output_bins)

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test


class MyDataSet:

    def __init__(self, f="/opt/project/mydata"):
        folder = Path(f)
        self.X = []
        for file in folder.iterdir():
            img = Image.open(file)
            img = img.convert("L")
            img = img.crop((0, 0, 220, 206))
            img = img.resize((64, 60), Image.NEAREST)
            self.X.append(np.array(img, dtype=np.float32))

        self.X = np.asarray(self.X)

        self.X_preprocessed = self.X / 255.0


class CommaAiDataSet:

    def __init__(self):
        with h5py.File("data/log/2016-06-08--11-46-01.h5", "r") as f:
            print(f.keys())

        with h5py.File("data/camera/2016-06-08--11-46-01.h5", "r") as f:
            print(f.keys())

if __name__ == '__main__':
    ds = CommaAiDataSet()

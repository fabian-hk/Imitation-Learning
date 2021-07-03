from typing import Tuple
import abc
import h5py
import numpy as np
from sklearn.utils import shuffle
from PIL import Image
from pathlib import Path

import utils
from visualization import visualize_angle_in_mp4, visualize_angle_in_gif


class DataSet:
    def __init__(self):
        self.X = []
        self.y = []

        self.X_resized = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.train_data_length = None

    @abc.abstractmethod
    def preprocessing(self, scale=None, output_bins=45):
        pass

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X_train, self.y_train

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X_test, self.y_test


class ReferenceDataSet(DataSet):
    def __init__(self, data_set_file="data/track_data_2.h5"):
        super(ReferenceDataSet, self).__init__()

        with h5py.File(data_set_file, "r") as f:
            for image, label in zip(f["images"], f["angles"]):
                self.X.append(image)
                self.y.append(label)

        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)

    def preprocessing(self, scale=None, output_bins=45):
        if scale is not None:
            # down scale images
            x_tmp = []
            for img in self.X:
                img = Image.fromarray(img)
                img = img.resize(scale, Image.NEAREST)
                x_tmp.append(np.array(img, dtype=np.float32))

            self.X_resized = np.asarray(x_tmp)
        else:
            self.X_resized = self.X

        # normalize images
        x_normalized = self.X_resized / 255.0

        # split data in train and test set and shuffle them
        split = round(len(x_normalized) * 0.7)
        self.train_data_length = split
        x_train = x_normalized[:split]
        y_train = self.y[:split]
        self.X_train, self.y_train = shuffle(x_train, y_train, random_state=16)

        self.X_test = x_normalized[split:]
        self.y_test = self.y[split:]

        # encode steering wheel angles
        self.y_train = utils.encode_angle(self.y_train, output_bins=output_bins)
        self.y_test = utils.encode_angle(self.y_test, output_bins=output_bins)


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


class CommaAiDataSet(DataSet):
    def __init__(self, data_set_path="/opt/project/"):
        super(CommaAiDataSet, self).__init__()

        with h5py.File(
            f"{data_set_path}data/commaai/camera/2016-06-08--11-46-01.h5", "r"
        ) as f:
            print(f["X"].shape)
            for x in f["X"]:
                self.X.append(x[2])

        with h5py.File(
            f"{data_set_path}data/commaai/log/2016-06-08--11-46-01.h5", "r"
        ) as f:
            steering_angle = f["steering_angle"]
            j = -1
            for i, v in enumerate(f["cam1_ptr"]):
                if v > j:
                    self.y.append(steering_angle[i])
                    j = v

        usable = (4800, 17000)

        self.X = self.X[usable[0] : usable[1]]
        self.y = self.y[usable[0] : usable[1]]

        self.X = np.asarray(self.X)
        print(f"X shape: {self.X.shape}")
        self.y = np.asarray(self.y)
        print(f"y shape: {self.y.shape}")

    def normalize_steering_angle(self):
        y_max = max(abs(np.min(self.y)), np.max(self.y))
        self.y = self.y / y_max
        self.y = self.y * 180

    def preprocessing(self, scale=None, output_bins=45):
        # down scale the images
        if scale is None:
            scale = (60, 64)

        X_tmp = []
        for x in self.X:
            img = Image.fromarray(x)
            img = img.resize(scale, Image.NEAREST)
            X_tmp.append(np.asarray(img, dtype=np.float32))

        self.X_resized = np.asarray(X_tmp)

        X_normalized = self.X_resized / 255.0

        # preprocess steering wheel angles
        self.normalize_steering_angle()

        # encode steering wheel angles
        self.y_encoded = utils.encode_angle(self.y, output_bins)

        # split data in train and test set and shuffle them
        split = round(len(X_normalized) * 0.6)
        self.train_data_length = split
        X_train = X_normalized[:split]
        y_train = self.y_encoded[:split]
        self.X_train, self.y_train = shuffle(X_train, y_train, random_state=16)

        self.X_test = X_normalized[split:]
        self.y_test = self.y_encoded[split:]


if __name__ == "__main__":
    ds = CommaAiDataSet("")
    ds.normalize_steering_angle()

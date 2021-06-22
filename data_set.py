from typing import Tuple
import h5py
import numpy as np
from sklearn.utils import shuffle


class DataSet:
    output_units = 45

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

    def _encode_angle(self, y: np.ndarray, train: bool) -> np.ndarray:
        y_tmp = []
        degree_per_bin = 180.0 / np.floor(self.output_units / 2.0)
        middle_bin = int(np.floor(self.output_units / 2.0))
        encoding = [0.1, 0.32, 0.61, 0.89, 1.0, 0.89, 0.61, 0.32, 0.1]
        for y_el in y:
            encoded_angle = np.zeros(self.output_units)
            angle_bin = middle_bin + int(np.round(y_el / degree_per_bin))
            if train:
                start = angle_bin - 4
                for i, e in enumerate(encoding):
                    encoded_angle[i + start] = e
            else:
                encoded_angle[angle_bin] = 1.0

            y_tmp.append(encoded_angle)

        return np.asarray(y_tmp)

    def decode_angle(self, y: np.ndarray) -> np.ndarray:
        degree_per_bin = 180.0 / np.floor(self.output_units / 2.0)
        middle_bin = int(np.floor(self.output_units / 2.0))
        bin = np.argmax(y, axis=1)
        return (bin - middle_bin) * degree_per_bin

    def preprocessing(self):
        # normalize images
        self.x = self.x / 255.0

        # split data in train and test set and shuffle them
        split = round(len(self.x)*0.7)
        self.train_data_length = split
        x_train = self.x[:split]
        y_train = self.y[:split]
        self.x_train, self.y_train = shuffle(x_train, y_train, random_state=16)

        self.x_test = self.x[split:]
        self.y_test = self.y[split:]

        # encode steering wheel angles
        self.y_train = self._encode_angle(self.y_train, True)
        self.y_test = self._encode_angle(self.y_test, True)

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test


if __name__ == '__main__':
    ds = DataSet()
    ds.preprocessing()
    pass

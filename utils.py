import numpy as np


def encode_angle(y: np.ndarray, output_bins=45) -> np.ndarray:
    y_encoded = []
    degree_per_bin = 180.0 / np.floor(output_bins / 2.0)
    middle_bin = int(np.floor(output_bins / 2.0))
    encoding = [0.1, 0.32, 0.61, 0.89, 1.0, 0.89, 0.61, 0.32, 0.1]
    for y_el in y:
        encoded_angle = np.zeros(output_bins)
        angle_bin = middle_bin + int(np.round(y_el / degree_per_bin))
        start = angle_bin - 4
        for i, e in enumerate(encoding):
            encoded_angle[i + start] = e
        y_encoded.append(encoded_angle)

    return np.asarray(y_encoded)


def decode_angle(y: np.ndarray, output_bins=45) -> np.ndarray:
    degree_per_bin = 180.0 / np.floor(output_bins / 2.0)
    middle_bin = int(np.floor(output_bins / 2.0))
    bin = np.argmax(y, axis=1)
    return (bin - middle_bin) * degree_per_bin

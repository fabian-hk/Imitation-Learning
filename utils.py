import numpy as np
import tensorflow as tf


def encode_angle(y: np.ndarray, output_bins) -> np.ndarray:
    y_encoded = []
    degree_per_bin = 180.0 / np.floor(output_bins / 2.0)
    middle_bin = int(np.floor(output_bins / 2.0))
    encoding = [0.1, 0.32, 0.61, 0.89, 1.0, 0.89, 0.61, 0.32, 0.1]
    for y_el in y:
        encoded_angle = np.zeros(output_bins)
        angle_bin = middle_bin + int(np.round(y_el / degree_per_bin))
        start = angle_bin - 4
        for i, e in enumerate(encoding):
            index = i + start
            if 0 <= index < len(encoded_angle):
                encoded_angle[index] = e
        y_encoded.append(encoded_angle)

    return np.asarray(y_encoded)


def decode_angle(y: np.ndarray, output_bins) -> np.ndarray:
    degree_per_bin = 180.0 / np.floor(output_bins / 2.0)
    middle_bin = int(np.floor(output_bins / 2.0))
    bin = np.argmax(y, axis=1)
    return (bin - middle_bin) * degree_per_bin


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    acc = 0.0
    for y0, y1 in zip(y_true, y_pred):
        y0_argmax = np.argmax(y0)
        y1_argmax = np.argmax(y1)
        if abs(y0_argmax - y1_argmax) < 2:
            acc += 1.0

    return acc / float(len(y_true))


def alvinn_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    # compute the argmax for each output sample
    y_true_argmax = tf.math.argmax(y_true, axis=1)
    y_pred_argmax = tf.math.argmax(y_pred, axis=1)

    # 1. Subtract the argmax values and take the absolute value
    # 2. Subtract 1 from this => all correct predictions are now either 0 or -1
    # 3. Use the relu function to set all correct predictions to 0
    # 4. Count the non zero values => inverse accuracy
    count_non_zero = tf.math.count_nonzero(
        tf.nn.relu(
            tf.subtract(tf.math.abs(tf.math.subtract(y_true_argmax, y_pred_argmax)), 1)
        )
    )

    # 1. Calculate the inverse accuracy
    # 2. Subtract it from 1 => accuracy
    return tf.math.subtract(
        tf.constant(1, dtype=tf.dtypes.float64),
        tf.math.divide(
            count_non_zero, tf.size(y_true_argmax, out_type=tf.dtypes.int64)
        ),
    )

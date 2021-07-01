import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from data_set import DataSet, CommaAiDataSet, ReferenceDataSet

from visualization import visualize_angle_in_gif
import utils


def train_and_evaluate(
    ds: DataSet = ReferenceDataSet(), input_size=None, output_bins=45, epochs=100
):
    tf.random.set_seed(1)
    np.random.seed(1)
    tf.config.experimental_run_functions_eagerly(True)

    # load data and preprocess them
    ds.preprocessing(scale=input_size, output_bins=output_bins)
    x_train, y_train = ds.get_train_data()

    input_shape = x_train[0].shape

    # define the model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(input_shape[0] * input_shape[1], activation=None),
            tf.keras.layers.Dense(29, activation=tf.keras.activations.sigmoid),
            tf.keras.layers.Dense(output_bins, activation=None),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[utils.alvinn_accuracy],
    )

    # train the model
    print("---- Training ----")
    history = model.fit(x_train, y_train, epochs=epochs)
    model.save("model/")

    plt.plot(history.history["loss"])
    plt.title("Loss over Epochs")
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Number of Epochs")
    plt.show()

    plt.plot(history.history["alvinn_accuracy"])
    plt.title("Accuracy over Epochs")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Epochs")
    plt.show()

    # test the model
    print("---- Testing ----")
    x_test, y_test = ds.get_test_data()
    model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)

    # visualize the prediction angle
    y_pred_degree = utils.decode_angle(y_pred, output_bins=output_bins)
    y_test_degree = utils.decode_angle(y_test, output_bins=output_bins)

    r = len(x_test)
    x = range(r)
    plt.plot(x, y_pred_degree, label="y_pred")
    plt.plot(x, y_test_degree, label="y_test")
    plt.legend()
    plt.title("Steering Angle (true and predicted) on the Test Data")
    plt.ylabel("Steering Angle")

    # visualize true and predicted angle in the images
    visualize_angle_in_gif(
        ds.X_resized[ds.train_data_length :],
        y_test_degree,
        y_pred_degree,
        f"visualization_{ds.__class__.__name__}_{epochs}_{input_size}_{output_bins}".replace(
            " ", ""
        ),
    )


if __name__ == "__main__":
    print(tf.__version__)

    train_and_evaluate(epochs=10, input_size=(60, 64), output_bins=65)

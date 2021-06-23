import tensorflow as tf
import matplotlib.pyplot as plt

from data_set import DataSet

from visualization import visualize_angle_in_gif
import utils


def train_and_evaluate(input_size=None, output_bins=45):
    tf.random.set_seed(1)

    # load data and preprocess them
    ds = DataSet()
    ds.preprocessing(scale=input_size, output_bins=output_bins)
    x_train, y_train = ds.get_train_data()

    input_shape = x_train[0].shape

    # define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(input_shape[0] * input_shape[1], activation=None),
        tf.keras.layers.Dense(29, activation=tf.keras.activations.sigmoid),
        tf.keras.layers.Dense(output_bins, activation=None)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.MeanSquaredError()
                  )

    # train the model
    print("---- Training ----")
    history = model.fit(x_train, y_train, epochs=100)
    model.save("model/")

    plt.plot(history.history["loss"])
    plt.title("Loss over Epochs")
    plt.ylabel("Mean Squared Error")
    plt.xlabel("#Epochs")
    plt.legend()

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
    visualize_angle_in_gif(ds.x[ds.train_data_length:], y_pred_degree, y_test_degree, f"visualization_{input_size}_{output_bins}".replace(" ", ""))

if __name__ == '__main__':
    train_and_evaluate(input_size=(30,32), output_bins=45)
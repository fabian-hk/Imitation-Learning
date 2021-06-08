import tensorflow as tf
import matplotlib.pyplot as plt

from data_set import DataSet

from visualization import draw_angle_in_video

# initialize tensorflow
print(f"Tensorflow version {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
tf.random.set_seed(12)

# load data and preprocess them
ds = DataSet()
ds.preprocessing()
x_train, y_train = ds.get_train_data()

# define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(60, 64)),
    tf.keras.layers.Dense(3840, activation=None),
    tf.keras.layers.Dense(29, activation=tf.keras.activations.sigmoid),
    tf.keras.layers.Dense(45, activation=None)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.MeanSquaredError()
              )

# train the model
print("---- Training ----")
model.fit(x_train, y_train, epochs=100)

# test the model
print("---- Testing ----")
x_test, y_test = ds.get_test_data()
model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

# visualize the prediction angle
y_pred_degree = ds.decode_angle(y_pred)
y_test_degree = ds.decode_angle(y_test)

r = len(x_test)
x = range(r)
plt.figure(0, (32, 10))
plt.plot(x, y_pred_degree, label="y_pred")
plt.plot(x, y_test_degree, label="y_test")
plt.legend()
plt.title("Steering Angle (true and predicted) on the Test Data")
plt.ylabel("Steering Angle")
plt.savefig("plot.png")

# visualize true and predicted angle in the images
draw_angle_in_video(x_test, y_pred_degree, y_test_degree)

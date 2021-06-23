import tensorflow as tf

from data_set import MyDataSet
import utils
from visualization import visualize_angle_in_gif

model = tf.keras.models.load_model("model/")

ds = MyDataSet("mydata/")

y_pred = model.predict(ds.X_preprocessed)

y_pred = utils.decode_angle(y_pred, output_bins=45)

visualize_angle_in_gif(ds.X, y_pred, fn="mydata_visualization", duration=500)

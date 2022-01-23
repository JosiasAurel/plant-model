import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# load trained model
model = keras.models.load_model("plant_model/")

image_path = "sample.jpg"
""" sample_image = image.load_img(image_path, target_size=(256, 256))
image_tensor = image.img_to_array(sample_image).reshape((256, 256, 3)) """

sample_image = Image.open("sample.jpg")
sample_image = np.asarray(sample_image)
# sample_image = np.reshape(sample_image, (32, 256, 3))
# sample_image = np.resize(sample_image, (32, 256, 3))
# sample_image = np.reshape(sample_image, (-1, 32, 256, 3))

image_tensor = tf.cast(sample_image, tf.float32)
image_tensor = tf.reshape(image_tensor, [-1, 256, 256, 3])
# print(sample_image, sample_image.shape)


# can we predict already ?
prediction = model.predict(image_tensor)
result = max(prediction[0])
print(result)
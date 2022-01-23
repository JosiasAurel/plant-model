import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import streamlit as st

# load trained model
model = keras.models.load_model("plant_model/")

class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 
'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

def make_prediction(image):
    loaded_image = Image.open(image)
    loaded_image = np.asarray(loaded_image)
    image_tensor = tf.cast(loaded_image, tf.float32)
    image_tensor = tf.reshape(image_tensor, [-1, 256, 256, 3])

    # can we predict already ?
    prediction = model.predict(image_tensor)
    prediction = np.array(prediction[0], dtype=np.int32).tolist()
    # print(prediction)
    def processed_result(prediction):
        max_val = -1
        max_idx = -1

        for idx, value in enumerate(prediction):
            if value > max_val:
                max_val = value
                max_idx = idx
        # print(max_val)
        return (" ".join(class_names[max_idx].split("_")), max_val)

    result = processed_result(prediction)
    return result 



st.title("Plant Disease Detection Model")
st.write("This is a sample model built as part of a bigger project run by Josias Aurel")
st.write("For any more information, email josias@josiasw.dev")

image = st.file_uploader("Upload Plant Image", type=["jpg", "jpeg", "png"])

if image is not None:
    result = make_prediction(image)
    st.metric(label="Prediction", value=result[0], delta=result[1])
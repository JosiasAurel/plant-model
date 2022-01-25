import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st
import onnxruntime as ort

class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
               'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


def make_prediction(session: ort.InferenceSession, image):
    loaded_image = Image.open(image)
    loaded_image = np.asarray(loaded_image)
    loaded_image = tf.image.resize(loaded_image, (256, 256))
    image_tensor = tf.cast(loaded_image, tf.float32)
    image_tensor = tf.reshape(image_tensor, [-1, 256, 256, 3])

    # can we predict already ?
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    prediction = session.run(output_names=[output_name], input_feed={
                             input_name: np.array(image_tensor)})
    # print(prediction)
    # print(np.max(prediction))
    # print(np.where(prediction == np.max(prediction))[-1])
    # print(prediction)

    def processed_result(prediction):
        max_val = np.max(prediction)
        max_idx = np.where(prediction == np.max(prediction))[-1][0]

        return (" ".join(class_names[max_idx].split("_")), int(abs(max_val)))

    result = processed_result(prediction)
    return result


st.title("Plant Disease Detection Model")
st.write("This is a sample model built as part of a bigger project run by Josias Aurel")
st.write("For any more information, email josias@josiasw.dev")

image = st.file_uploader("Upload Plant Image", type=["jpg", "jpeg", "png"])

if image is not None:
    session = ort.InferenceSession("plant_model.onnx")
    result = make_prediction(session, image)
    st.metric(label="Prediction", value=result[0], delta=result[1])

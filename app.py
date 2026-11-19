
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Plant Disease Detection System")

st.write(
    "Upload a clear plant leaf image to identify "
    "its condition using an AI-based machine learning model."
)

# ---------------------------------
# MODEL LOADING
# ---------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "plant_disease_model (2).keras",
        compile=False
    )

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ---------------------------------
# ORIGINAL 38 CLASS NAMES
# ---------------------------------

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___healthy",
    "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___healthy",
    "Corn___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

# ---------------------------------
# DISEASE INFORMATION
# ---------------------------------

def get_disease_info(label):

    plant, condition = label.split("___", 1)

    condition = condition.replace("_", " ")
    condition = condition.replace(",", "")
    condition = condition.replace("(", "").replace(")", "")

    if "healthy" in label.lower():
        information = (
            f"The model classified this {plant} leaf as healthy. "
            "No listed disease was identified as the top prediction."
        )
        prevention = (
            "Continue regular monitoring, proper watering, "
            "good sanitation and balanced plant nutrition."
        )
    else:
        information = (
            f"This prediction corresponds to {condition} in {plant}. "
            "The model identifies visual patterns associated with "
            "this disease. Similar symptoms can occur for other reasons."
        )
        prevention = (
            "Remove and dispose of affected plant material appropriately. "
            "Keep foliage dry where possible, provide good air circulation, "
            "sanitize gardening tools and monitor nearby plants. "
            "Consult an agricultural expert before applying treatment."
        )

    return information, prevention

# ---------------------------------
# IMAGE PREDICTION
# ---------------------------------

def predict_disease(image):

    image = image.convert("RGB")
    image = image.resize((160, 160))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    top_indices = np.argsort(predictions)[-3:][::-1]

    return [
        (class_names[i], float(predictions[i]) * 100)
        for i in top_indices
    ]

# ---------------------------------
# IMAGE UPLOAD
# ---------------------------------

uploaded_file = st.file_uploader(
    "📷 Upload Plant Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Plant Leaf",
        use_container_width=True
    )

    if st.button("🔍 Predict Disease", type="primary"):

        with st.spinner("Analyzing leaf image..."):

            results = predict_disease(image)

        label, confidence = results[0]

        plant, condition = label.split("___", 1)

        information, prevention = get_disease_info(label)

        st.success("Prediction completed!")

        st.subheader("🌿 Prediction Result")

        st.write("**Plant:**", plant)

        st.write(
            "**Predicted Condition:**",
            condition.replace("_", " ")
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("📊 Top 3 Predictions")

        for name, score in results:

            st.write(
                f"**{name.replace('___', ' - ')}**"
            )

            st.progress(
                min(max(score / 100, 0.0), 1.0)
            )

            st.caption(f"{score:.2f}%")

        st.subheader("📖 Disease Information")
        st.write(information)

        st.subheader("🛡️ Prevention")
        st.write(prevention)

        st.warning(
            "AI prediction, not a confirmed agricultural diagnosis. "
            "Predictions may be unreliable for unclear images, "
            "unseen diseases or plants outside the training classes."
        )

st.divider()

st.caption(
    "Plant Disease Detection | MobileNetV2 | "
    "PlantVillage | 38 Classes"
  )

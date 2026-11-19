
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Plant Disease Detection System")

st.write(
    "Upload a clear plant leaf image to identify its disease "
    "using an AI-based machine learning model."
)

# ==========================================
# MODEL LOADING
# ==========================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "plant_disease_model (2).keras",
        compile=False
    )


@st.cache_resource
def load_validation_model():

    return tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=True
    )


try:
    model = load_model()
    validation_model = load_validation_model()

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ==========================================
# ORIGINAL 38 CLASS NAMES
# ==========================================

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


# ==========================================
# NON-PLANT IMAGE VALIDATION
# ==========================================

NON_PLANT_LABELS = {
    "web_site", "menu", "envelope", "comic_book",
    "baseball", "ballplayer", "soccer_ball",
    "dog", "cat", "car", "truck", "bus",
    "person", "bicycle", "motorcycle", "airliner"
}


# ==========================================
# DISEASE INFORMATION
# ==========================================

def get_disease_info(label):

    plant, condition = label.split("___", 1)

    condition = condition.replace("_", " ")
    condition = condition.replace(",", "")
    condition = condition.replace("(", "")
    condition = condition.replace(")", "")

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


# ==========================================
# FINAL PREDICTION FUNCTION
# ==========================================

def final_predict(image):

    if image is None:
        return None, "Please upload an image."

    try:

        image = image.convert("RGB")

        # STAGE 1: VALIDATE UPLOADED IMAGE

        check_image = image.resize((224, 224))

        arr = np.array(
            check_image,
            dtype=np.float32
        )

        arr = np.expand_dims(arr, axis=0)

        arr = tf.keras.applications.mobilenet_v2.preprocess_input(
            arr
        )

        validation_output = validation_model.predict(
            arr,
            verbose=0
        )

        top_results = decode_predictions(
            validation_output,
            top=5
        )[0]

        for _, label, score in top_results:

            if label in NON_PLANT_LABELS and score >= 0.30:

                return None, (
                    "Image not found. Please upload a plant leaf image."
                )

        # STAGE 2: PLANT DISEASE PREDICTION

        disease_image = image.resize((160, 160))

        disease_arr = np.array(
            disease_image,
            dtype=np.float32
        )

        disease_arr = (
            tf.keras.applications.mobilenet_v2.preprocess_input(
                disease_arr
            )
        )

        disease_arr = np.expand_dims(
            disease_arr,
            axis=0
        )

        predictions = model.predict(
            disease_arr,
            verbose=0
        )[0]

        # TOP 3 PREDICTIONS

        top_indices = np.argsort(
            predictions
        )[-3:][::-1]

        results = []

        for index in top_indices:

            results.append((
                class_names[index],
                float(predictions[index]) * 100
            ))

        confidence = results[0][1]

        # CONFIDENCE VALIDATION

        if confidence < 70:

            return None, (
                "Image not found or unclear. "
                "Please upload a clear plant leaf image."
            )

        return results, None

    except Exception as e:

        return None, f"Prediction error: {str(e)}"


# ==========================================
# IMAGE UPLOAD INTERFACE
# ==========================================

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

    if st.button(
        "🔍 Predict Disease",
        type="primary"
    ):

        with st.spinner("Analyzing leaf image..."):

            results, error = final_predict(image)

        if error:

            st.error(error)

        else:

            label, confidence = results[0]

            plant, condition = label.split("___", 1)

            information, prevention = get_disease_info(label)

            # PREDICTION RESULT

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

            # TOP 3 PREDICTIONS

            st.subheader("📊 Top 3 Predictions")

            for name, score in results:

                st.write(
                    f"**{name.replace('___', ' - ')}**"
                )

                st.progress(
                    min(max(score / 100, 0.0), 1.0)
                )

                st.caption(
                    f"{score:.2f}%"
                )

            # INFORMATION

            st.subheader("📖 Disease Information")

            st.write(information)

            # PREVENTION

            st.subheader("🛡️ Prevention")

            st.write(prevention)

            # DISCLAIMER

            st.warning(
                "AI prediction, not a confirmed agricultural diagnosis. "
                "Predictions may be unreliable for unclear images, "
                "unseen diseases or plants outside the training classes."
            )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Plant Disease Detection | MobileNetV2 | "
    "PlantVillage Dataset | 38 Classes"
)

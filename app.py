import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# -------------------------------
# Load Model and Label Names
# -------------------------------
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("leaf_species_model_epoch_2.keras")
    with open("species_labels-epoch-2.json", "r") as f:
        label_names = json.load(f)
    if isinstance(label_names, dict):
        label_names = {int(k): v for k, v in label_names.items()}
    return model, label_names

model, label_names = load_model_and_labels()
IMG_SIZE = 96

# -------------------------------
# Preprocess Image
# -------------------------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, IMG_SIZE, IMG_SIZE, 3)
    return img

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌿 Leaf Classification App")
st.write("Upload a leaf image and let the trained model predict its species!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])



if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Leaf Image", use_container_width=True)

    # Spinner while predicting
    with st.spinner("🔎 Analyzing leaf... please wait..."):
        img = preprocess_image(uploaded_file)
        pred = model.predict(img)
        pred_class = int(np.argmax(pred[0]))
        confidence = float(np.max(pred[0]) * 100)

    # Show results
    st.subheader("🌱 Prediction Results:")
    st.write(f"**Predicted Class Index:** {pred_class}")
    st.write(f"**Predicted Class Name:** {label_names[pred_class].split('_')[0]}")

    # Confidence progress bar
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.progress(int(confidence))

    # Final message
    st.success("✅ Analysis complete!")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 14px;'>
        🌱 Powered by TensorFlow | Made with ❤️ by Rakesh, Nishanth, Josika <br>
        <a href="https://github.com/rakesh001n/Leaf-Classification---precision-Agriculture" target="_blank">
            📂 View Source Code on GitHub
        </a>
        <br>
        <a href="https://www.kaggle.com/code/rockybhai001n/leaf-classification-precision-agriculture/notebook" target="_black">
        📒 Kaggle Notebook
        </a>
    </div>
    """,
    unsafe_allow_html=True
)


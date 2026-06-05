from rag_module import rag_chatbot
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"
import cv2
import streamlit as st
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt

import io
from PIL import Image
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

def predict_and_explain(image):
    result = model.predict(image, verbose=False)[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return "No egg detected", "N/A", None

    idx = torch.argmax(boxes.conf).item()
    species_id = int(boxes.cls[idx].item())
    species = model.names[species_id]
    confidence = float(boxes.conf[idx].item())

    box = boxes.xyxy[idx].int().tolist()
    crop = image[box[1]:box[3], box[0]:box[2]]

    if crop.size == 0:
        return species, f"{confidence:.2f}", None

    egg = cv2.resize(crop, (128, 128)).astype(np.float32) / 255.0
    TARGET_CLASS_ID = species_id

    def yolo_score(images):
        if images.ndim == 3:
            images = images[np.newaxis, ...]
        outputs = []
        model.model.eval()
        with torch.no_grad():
            for img in images:
                x = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
                preds = model.model(x)[0]
                probs = preds[..., -model.model.nc:].sigmoid()
                outputs.append(probs[..., TARGET_CLASS_ID].max().item())
        return np.array(outputs)

    masker = shap.maskers.Image("inpaint_telea", egg.shape)
    explainer = shap.Explainer(yolo_score, masker)
    shap_values = explainer(egg[np.newaxis, ...], max_evals=50)

    plt.close("all")
    shap.image_plot(shap_values, egg[np.newaxis, ...], show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    shap_image = Image.open(buf)

    return species, f"{confidence * 100:.0f}%", shap_image


st.set_page_config(page_title="Nest Best Thing", layout="wide")

st.title("🪺 Nest Best Thing (Explainable AI)")

st.markdown(
"""
This tool helps identify bird egg species using AI.

Upload an egg image to get:
- 🧠 Species prediction  
- 📊 Confidence score  
- 🔍 SHAP explanation  
"""
)

st.divider()

# Layout with columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Egg Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "png", "jpeg"],
        help="Use a clear image with minimal background"
    )

with col2:
    st.subheader("ℹ️ Supported Scope")
    st.markdown("""
    - Model supports **21 bird species**
    - Best results with:
        - Good lighting  
        - Centered egg  
        - Minimal shadows  
    """)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    species, confidence, shap_image = predict_and_explain(image_np)

    st.write(f"**Predicted Species:** {species}")
    st.write(f"**Confidence:** {confidence}")

    if shap_image:
        st.image(shap_image, caption="SHAP Explanation")


st.divider()
st.header("📚 IBIS Assistant (RAG Chatbot)")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("Ask a question about the eggs or species")

if user_input:
    answer = rag_chatbot(user_input)

    # Save chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("IBIS", answer))

# Display chat history
for role, text in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {text}")
    else:
        st.markdown(f"**🤖 IBIS:** {text}")



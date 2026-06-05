import cv2
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import gradio as gr
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
    shap_values = explainer(egg[np.newaxis, ...], max_evals=200)

    plt.close("all")
    shap.image_plot(shap_values, egg[np.newaxis, ...], show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    shap_image = Image.open(buf)

    return species, f"{confidence:.2f}", shap_image

demo = gr.Interface(
    fn=predict_and_explain,
    inputs=gr.Image(type="numpy", label="Upload Egg Image"),
    outputs=[
        gr.Text(label="Predicted Species"),
        gr.Text(label="Confidence Score"),
        gr.Image(label="SHAP Explanation"),
    ],
    title="Nest Best Thing (Explainable AI)",
    description="Upload an egg image to predict the species and view a SHAP explanation."
)

demo.launch()

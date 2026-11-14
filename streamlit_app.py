from pathlib import Path
import streamlit as st
from PIL import Image
from model_utils import load_model, looks_like_xray, predict_with_gradcam

BASE = Path(__file__).parent

st.set_page_config(page_title="Pediatric Pneumonia Detector", page_icon="🫁", layout="wide")


@st.cache_resource          # load the model once, not on every click
def get_model():
    return load_model(BASE / "resnet18_chestcrop_fp16.pth")


st.title("🫁 Pediatric Pneumonia Detector with Grad-CAM")
st.markdown("Upload a **pediatric chest X-ray** (front view) to see the model's prediction "
            "and a **Grad-CAM heatmap** showing which part of the chest influenced the decision.")
st.warning("⚠️ Educational project only. This is not a medical device and must not be used for diagnosis.")

examples = sorted((BASE / "examples").glob("*.jpeg"))
uploaded = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])
choice = st.selectbox("...or try an example", ["(none)"] + [p.name for p in examples])

image = None
if uploaded is not None:
    image = Image.open(uploaded)
elif choice != "(none)":
    image = Image.open(BASE / "examples" / choice)

if image is not None:
    if not looks_like_xray(image):
        st.error("This doesn't look like a chest X-ray (it contains colour). "
                 "Please upload a grayscale, front-view chest X-ray.")
    else:
        with st.spinner("Analysing..."):
            probs, overlay = predict_with_gradcam(get_model(), image)
        label = max(probs, key=probs.get)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Your X-ray")
            st.image(image.convert("RGB"), width=380)
        with col2:
            st.subheader(f"Prediction: {label}")
            for name, p in probs.items():
                st.progress(p, text=f"{name}: {p:.0%}")
            st.image(overlay, caption="Grad-CAM: where the model looked (chest crop)", width=380)

with st.expander("About the model and its limitations"):
    st.markdown("""
**Model:** ResNet-18 pretrained on ImageNet, fine-tuned on chest X-rays cropped to the chest region.
Class-weighted loss, X-ray-appropriate augmentation (no horizontal flips), patient-level train/validation split.

**Test set (624 images):** accuracy 90.7%, pneumonia recall 99.5%, normal recall 76.1%.

**Known limitations**
- Trained only on X-rays of young children from a single hospital; it will not generalise reliably to adults or other hospitals.
- It wrongly flags about 1 in 4 healthy children as pneumonia.
- Grad-CAM analysis suggested the model partly relies on cues tied to patient age and positioning, not only lung appearance.
- A colour check rejects colour photos, but grayscale images that aren't chest X-rays still get a confident (and meaningless) answer.

**Dataset:** Kermany et al., 2018, *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, Mendeley Data (CC BY 4.0).
""")

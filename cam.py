# ═══════════════════════════════════════════════════════════════
#  Deep Learning Object Detection Demo
#  YOLOv5 · Streamlit · Webcam · Bounding Boxes · Confidence
#  Run with: streamlit run object_detection_demo.py
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import torch
import numpy as np
from PIL import Image

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(page_title="Deep Learning Object Detection Demo", layout="centered")
st.title("🎯 Deep Learning Object Detection Demo")
st.caption("Powered by YOLOv5 · Point your camera at anything and hit capture.")

# ── Load YOLOv5 model (cached so it loads only once) ────────────
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.conf = 0.40   # minimum confidence threshold
    model.iou  = 0.45   # NMS IoU threshold
    return model

with st.spinner("Loading YOLOv5s model…"):
    model = load_model()
st.success("✅ Model ready — capture a photo to detect objects.", icon="✅")

st.divider()

# ── Camera input ─────────────────────────────────────────────────
photo = st.camera_input("📷 Take a picture")

if photo is not None:

    # ── Convert captured image → PIL → NumPy ────────────────────
    pil_image  = Image.open(photo).convert("RGB")
    img_array  = np.array(pil_image)

    # ── Run inference ────────────────────────────────────────────
    with st.spinner("Running object detection…"):
        results = model(img_array)

    # ── Annotated image (YOLOv5 built-in renderer) ───────────────
    annotated = np.array(results.render()[0])   # BGR numpy array with boxes drawn
    st.image(annotated, caption="Detection Results", use_container_width=True)

    st.divider()

    # ── Parse detections ─────────────────────────────────────────
    # results.pandas().xyxy[0] → DataFrame with columns:
    # xmin ymin xmax ymax confidence name class
    df = results.pandas().xyxy[0]

    if df.empty:
        st.warning("No objects detected. Try better lighting or move closer.")
    else:
        # ── Summary section ───────────────────────────────────────
        total_objects  = len(df)
        unique_classes = df["name"].unique().tolist()

        col1, col2 = st.columns(2)
        col1.metric("📦 Total Objects Detected", total_objects)
        col2.metric("🏷️ Unique Classes", len(unique_classes))

        st.markdown("**Detected classes:** " + " · ".join(
            f"`{cls}`" for cls in unique_classes
        ))

        st.divider()

        # ── Per-object summary ────────────────────────────────────
        st.subheader("🔍 Detected Objects")
        for _, row in df.iterrows():
            conf_pct = f"{row['confidence']:.0%}"
            st.markdown(f"- **{row['name'].capitalize()}** — {conf_pct}")

        st.divider()

        # ── Detailed table ────────────────────────────────────────
        st.subheader("📋 Detection Details")
        display_df = df[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]].copy()
        display_df.columns = ["Object Name", "Confidence", "x_min", "y_min", "x_max", "y_max"]
        display_df["Confidence"]  = display_df["Confidence"].map("{:.2%}".format)
        display_df["Bounding Box"] = display_df.apply(
            lambda r: f"({int(r.x_min)}, {int(r.y_min)}) → ({int(r.x_max)}, {int(r.y_max)})",
            axis=1,
        )
        st.dataframe(
            display_df[["Object Name", "Confidence", "Bounding Box"]],
            use_container_width=True,
            hide_index=True,
        )

# ── Footer ───────────────────────────────────────────────────────
st.divider()
st.caption("YOLOv5s · PyTorch · Streamlit · COCO-pretrained (80 object classes)")
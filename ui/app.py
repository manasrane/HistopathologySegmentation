import streamlit as st
import requests
import base64
from PIL import Image
import io
import numpy as np

st.title("Histopathology Gland Segmentation")

file = st.file_uploader("Upload Histopathology Image", type=["png", "jpg", "jpeg"])

if file:
    st.image(file, caption="Original Image", use_column_width=True)

    # Send to API
    response = requests.post(
        "http://localhost:8000/segment",
        files={"file": file}
    )

    if response.status_code == 200:
        data = response.json()
        mask_data = base64.b64decode(data["mask"])
        mask_img = Image.open(io.BytesIO(mask_data))

        # Overlay
        original = Image.open(file).convert("RGB").resize(mask_img.size)
        overlay = Image.blend(original, Image.new("RGB", original.size, (255, 0, 0)), 0.5)
        mask_array = np.array(mask_img) > 128
        overlay_array = np.array(overlay)
        overlay_array[mask_array] = [255, 0, 0]  # Red for glands
        overlay_img = Image.fromarray(overlay_array)

        col1, col2 = st.columns(2)
        with col1:
            st.image(mask_img, caption="Predicted Mask", use_column_width=True)
        with col2:
            st.image(overlay_img, caption="Overlay", use_column_width=True)

        st.success("Segmentation completed!")
    else:
        st.error("API request failed")
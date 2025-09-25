import streamlit as st
import numpy as np
from PIL import Image, ImageDraw

# --- Mock prediction function ---
def mock_runway_detection(image):
    # Create a blank mask (same size as input)
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    # Draw a fake runway rectangle in the center
    w, h = image.size
    draw.rectangle([(w//3, h//2), (2*w//3, h-20)], fill=255)
    # Fake anchor points (corners of the rectangle)
    anchor_points = [
        (w//3, h//2), (2*w//3, h//2), (w//3, h-20), (2*w//3, h-20)
    ]
    orientation = 90  # degrees (mock value)
    return mask, anchor_points, orientation

st.title("Runway Detection Using Computer Vision")

uploaded_file = st.file_uploader("Upload an aircraft landing image", type=["jpg", "png", "jpeg"])
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400&display=swap');
    html, body, [class*='css'] {
        font-family: 'Roboto', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)


if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)
    
    # Use mock prediction
    mask, anchor_points, orientation = mock_runway_detection(image)
    
    # Overlay mask on image
    overlay = image.copy()
    overlay.paste((255,0,0), mask=mask)
    st.image(overlay, caption="Detected Runway (Mock)", use_column_width=True)
    
    # Show anchor points
    st.write("**Anchor Points (mock):**", anchor_points)
    st.write(f"**Orientation (mock):** {orientation}°")
else:
    st.info("Please upload an image to test the runway detection.")

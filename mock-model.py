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
import streamlit as st
from datetime import datetime

# --- Set background image using CSS ---
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('https://images.pexels.com/photos/62623/wing-plane-flying-airplane-62623.jpeg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        min-height: 100vh;
        opacity: 0.5 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Artistic, centered title and tagline ---
st.markdown(
    """
    <h1 style="text-align: center; color: #22223b; font-family: 'Segoe UI', 'Roboto', sans-serif; font-size: 3em; letter-spacing: 2px; text-shadow: 2px 2px 8px #b5b5b5;">
        Runway Detection Using Computer Vision
    </h1>
    <h3 style="text-align: center; color: #4B8BBE; font-family: 'Segoe UI', 'Roboto', sans-serif; font-weight: 400;">
        Enhancing aviation safety with AI-powered runway detection.
    </h3>
    """,
    unsafe_allow_html=True
)

# --- Show current date and time ---
now = datetime(2025, 9, 26, 2, 0)  # Friday, September 26, 2025, 2 AM IST
st.markdown(
    f"<p style='text-align:center; color:#555; font-size:1.1em;'>🗓️ {now.strftime('%A, %B %d, %Y, %I:%M %p IST')}</p>",
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

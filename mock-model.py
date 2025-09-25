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



# --- Set background image using CSS ---
import streamlit as st

st.markdown(
    """
    <style>
    /* Main app background with overlay */
    .stApp {
        position: relative;
        background: url("https://images.pexels.com/photos/1436697/pexels-photo-1436697.jpeg")
                    no-repeat center center fixed;
        background-size: cover;
    }

    /* Add a dark overlay only on the background */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0,0,0,0.5); /* 0.5 darkness */
        z-index: 0;
    }

    /* Ensure app content stays above the overlay */
    .stApp > div {
        position: relative;
        z-index: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Artistic, centered title and tagline ---
st.markdown(
    """
    <h1 style="display: block; margin-left: auto; margin-right: auto; text-align: center; white-space: nowrap; color: #FFFFFF; font-family: 'Abril Fatface'; font-size: 40px;">
        Runway Detection Using Computer Vision
    </h1>
    <h3 style="text-align: center; color: #B3EFFF; font-family: 'Teko'; font-weight: 400;">
        Enhancing aviation safety with AI-powered runway detection.
    </h3>
    """,
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("Upload a runway image", type=["jpg", "png", "jpeg"])


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

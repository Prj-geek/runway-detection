import streamlit as st
st.set_page_config(layout="wide")
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


# --- Artistic, centered title and tagline ---
st.markdown(
    """
    <h1 style="text-align: center; white-space: nowrap; color: #FFFFFF; font-family: 'Abril Fatface'; font-size: 2.5em;">
        Runway Detection Using Computer Vision
    </h1>
    <h4 style="display: block; margin-left: auto; margin-right: auto; text-align: center; white-space: nowrap; color: #B3EFFF; font-family: 'Teko'; font-weight: 400;">
        Enhancing aviation safety with AI-powered runway detection.
    </h4>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <style>
    /* Center the label text of file uploader */
    .stFileUploader label {
        display: flex;
        justify-content: center;
        font-weight: bold; /* optional */
        text-align: center;
        width: 100%;
        font-size: 2em
    
    }

    /* Center the entire uploader widget itself */
    .stFileUploader {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.subheader("Demo: Try Runway Detection")
uploaded_file = st.file_uploader("Upload an aerial or drone image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Open and process image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    mask, anchor_points, orientation = mock_runway_detection(img)
    st.subheader("Detected Runway Segmentation (Mock)")
    st.image(mask, caption="Mock Segmentation Output", use_column_width=True)
    st.write("Anchor Points (Mock):", anchor_points)
    st.write("Runway Orientation (Mock):", orientation)
else:
    st.info("Please upload an image to test runway detection.")

st.caption("Built by Team for Hackathon - MIT Bengaluru")


# --- Set background image using CSS ---

st.markdown(
    """
    <style>
    .main > div {
        padding-left: 0rem;
        padding-right: 0rem;
    }
    .block-container {
        padding-left: 0rem;
        padding-right: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <style>
    /* Main app background with overlay */
    .stApp {
        position: relative;
        background: url("https://images.pexels.com/photos/615060/pexels-photo-615060.jpeg")
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




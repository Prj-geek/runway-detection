import base64
import io

from PIL import Image, ImageDraw

import streamlit as st

# --- Mock runway detection function ---
def mock_runway_detection(image: Image.Image) -> Image.Image:
    """
    Pretend to detect runway and highlight it on the image.
    This is a placeholder for your model integration.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    # Draw a mock "runway" as a yellow rectangle in the center
    w, h = img.size
    left = w // 4
    top = h // 3
    right = w * 3 // 4
    bottom = h * 2 // 3
    draw.rectangle([left, top, right, bottom], outline="yellow", width=8)
    return img

# --- Sidebar layout with tabs ---
def sidebar():
    with st.sidebar:
        st.markdown("# Navigation")
        tabs = ["Runway Detection", "About", "Feedback"]
        selected_tab = st.radio("Go to", tabs)
        st.markdown("---")
        if selected_tab == "About":
            st.markdown(
                "This tool allows you to upload an aerial image and detect runways automatically. "
                "The model highlights detected runways for visualization."
            )
            st.markdown(
                "This project is open-source and a work in progress. "
                "You can contribute to the project on [GitHub](https://github.com/Prj-geek/runway-detection) "
                "with your feedback and suggestions 💡"
            )
        elif selected_tab == "Feedback":
            st.markdown(
                "## Feedback\n"
                "We welcome your feedback and suggestions! Please open an issue or pull request on [GitHub](https://github.com/Prj-geek/runway-detection)."
            )
        st.markdown("---")
    return selected_tab

# --- Main App ---
st.set_page_config(page_title="Runway Detection Tool",
                   layout="centered",
                   initial_sidebar_state="expanded")

st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)  # Removes Page Navigation

st.title("Runway Detection 🛬")

selected_tab = sidebar()

if selected_tab == "Runway Detection":
    st.header("Upload your aerial image")
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image

            with st.spinner("Detecting runway..."):
                result_img = mock_runway_detection(image)
                st.session_state.processed_image = result_img

            st.success("Runway detected and highlighted below!")
            st.image(result_img, caption="Detected Runway", use_column_width=True)
        except Exception as e:
            st.error(f"Error processing the image: {e}")

elif selected_tab == "About":
    st.header("About Runway Detection Tool")
    # Content handled in sidebar

elif selected_tab == "Feedback":
    st.header("Feedback")
    # Content handled in sidebar

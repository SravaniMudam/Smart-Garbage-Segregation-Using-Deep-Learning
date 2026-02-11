import streamlit as st
from PIL import Image
import urllib.request
from utils import predict_image

st.set_page_config(
    page_title="Smart Garbage Segregation",
    layout="centered"
)

st.markdown("""
<h1 style="text-align:center;">♻️ Smart Garbage Segregation</h1>
<h3 style="text-align:center;color:teal;">Waste Classification App</h3>
""", unsafe_allow_html=True)

option = st.selectbox(
    "Select image input method:",
    ("Select", "Upload from device", "Upload via URL")
)

image = None

if option == "Upload from device":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if file:
        image = Image.open(file)

elif option == "Upload via URL":
    url = st.text_input("Enter image URL")
    if url:
        try:
            image = Image.open(urllib.request.urlopen(url))
        except:
            st.error("Invalid image URL")

if image is not None:
    st.image(image, width=300, caption="Uploaded Image")

    if st.button("Predict"):
        with st.spinner("Classifying waste..."):
            result = predict_image(image)

        st.success(f"🗑️ Type: **{result.upper()}**")

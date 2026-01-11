import streamlit as st
import requests
from PIL import Image
import io
from os import getenv

# Read the url from environment variable
API_URL = getenv("API_URL", "http://127.0.0.1:8000")

# create the title and the subtitle
st.title("Cats & Dogs Classifier")
st.write("Upload an image to classify it using either a pretrained or a finetuned model.")

# create the image uplader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# Cretate the model selector
model_option = st.selectbox(
    "Select Model",
    ("Hugging Face ViT-Pretrained", "Mobilenet_v3_small-Finetuned")
)

if uploaded_file is not None:
    
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='The Uploaded Image', width="stretch")
    
    if st.button('Classify'):
        with st.spinner('Analyzing...'):
            
            # convert image to bytes in order to send it back to API
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format)
            img_byte_arr = img_byte_arr.getvalue()
            
            # call FastAPI Endpoint
            files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
            data = {'model_type': model_option.split("-")[0]}
            
            try:
                # assuming FastAPI runs on localhost:8000
                response = requests.post(f"{API_URL}/predict", files=files, data=data)
                result = response.json()
                
                # return the predicted class along with the prediction confidence
                st.success(f"Prediction: **{result['prediction']}**")
                st.info(f"Confidence: {result['confidence']:.2%}")
                
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
import sys
sys.modules['torch.classes'] = None # Exclude to prevent PyTorch/Streamlit compatibility error

import streamlit as st
import torch

from utils import load_model, preprocess_image, predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pytorch Model Prediction
def on_predict_click(image: str):
    # Load the pre-trained model
    model = load_model("../swin_model_10_epochs.pth", device=device)
    
    # Preprocess the image
    image_tensor = preprocess_image(image)

    # Predict the class
    return predict(model=model, image_tensor=image_tensor, device=device)

# Sidebar
st.sidebar.title("Dashboard")    
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Predict"])

# Page ideas
# various models?, metrics?

# Home Page
if(app_mode == "Home"):
    st.header("Plant Disease Recognition System") 
    image_path = "images/home_page.jpeg"
    st.image(image_path, caption="Home Page", use_container_width=True)
    st.markdown(
        """
        This is a plant disease recognition system that helps farmers and agricultural professionals identify diseases in plants using machine learning techniques. 
        The system uses a pre-trained model to analyze images of plants and provide accurate predictions of potential diseases.
        """
    )   

# About Page
elif(app_mode == "About"):
    st.header("About")
    st.markdown(
        """
        ### About the Dataset
        This version of the PlantVillage dataset is was downloaded from TensorFlow Datasets. 
        It can be found at: https://www.tensorflow.org/datasets/catalog/plant_village
        ### Content: 54,305 images across 38 classes
        1. Training: 70%
        2. Validation: 20%
        3. Testing: 10%
        """
    )

# Predict Page
elif(app_mode == "Predict"):
    st.header("Predict")
    st.markdown(
        """
        Upload an image of a plant leaf to predict its health status.
        """
    )
    
    # File uploader
    test_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        if(st.button("Show Image")):
            st.image(test_image, caption='Uploaded Image', use_container_width=True)
       
        if(st.button("Predict")):
            with st.spinner("Thinking..."):
                # Call the predict function
                prediction = on_predict_click(test_image)
                print(prediction)
                st.write("Prediction:")
                st.success(prediction)

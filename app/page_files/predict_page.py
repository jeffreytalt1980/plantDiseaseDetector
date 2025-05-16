import sys
sys.modules['torch.classes'] = None # Exclude to prevent PyTorch/Streamlit compatibility error

import streamlit as st
import torch
import os

from utils import load_model, preprocess_image, predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pytorch Model Prediction
def on_predict_click(image: str):
    # Load the pre-trained model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../swin_model_plant_disease_detector.pth"))
    model = load_model(model_path, device=device)
    
    # Preprocess the image
    image_tensor = preprocess_image(image)

    # Predict the class
    return predict(model=model, image_tensor=image_tensor, device=device)

def predict_page():
    st.header("Predict")
    st.markdown(
        """
        Upload image files (png, jpg, or jpeg) of plant leaves to predict their condition.
        """
    )
    
    # File uploader
    test_images = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if len(test_images) is 1:
        st.image(test_images[0], caption='Uploaded Image', use_container_width=True)
       
        if(st.button("Predict")):
            with st.spinner("Thinking..."):
                # Call the predict function
                prediction, confidence = on_predict_click(test_images[0])
                st.write("Prediction:")
                st.success(f"The model predicts that this is \"{prediction}\" with a confidence of {confidence*100:.2f}%.")
    elif len(test_images) > 1:
        if(st.button("Predictions")):
            with st.spinner("Thinking..."):
                # Call the predict function
                predictions = []
                for test_image in test_images:
                    prediction, confidence = on_predict_click(test_image)
                    predictions.append((test_image, prediction, confidence))
                
                st.write("Predictions:")
                for test_image, prediction, confidence in predictions:
                    st.image(test_image, caption='Uploaded Image', use_container_width=True)
                    st.success(f"The model predicts that \"{test_image.name}\" is \"{prediction}\" with a confidence of {confidence*100:.2f}%.")

    st.session_state.test_images = None
import sys
sys.modules['torch.classes'] = None # Exclude to prevent PyTorch/Streamlit compatibility error

import streamlit as st
import torch

from utils import load_model, preprocess_image, predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pytorch Model Prediction
def on_predict_click(image: str):
    # Load the pre-trained model
    model = load_model("../swin_model_10_epochs2.pth", device=device)
    
    # Preprocess the image
    image_tensor = preprocess_image(image)

    # Predict the class
    return predict(model=model, image_tensor=image_tensor, device=device)

def predict_page():
    st.header("Predict")
    st.markdown(
        """
        Upload an image file (png, jpg, or jpeg) of a plant leaf to predict its condition.
        """
    )
    
    # File uploader
    test_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        st.image(test_image, caption='Uploaded Image', use_container_width=True)
       
        if(st.button("Predict")):
            with st.spinner("Thinking..."):
                # Call the predict function
                prediction, confidence = on_predict_click(test_image)
                st.write("Prediction:")
                st.success(f"The model predicts that this is \"{prediction}\" with a confidence of {confidence*100:.2f}%.")

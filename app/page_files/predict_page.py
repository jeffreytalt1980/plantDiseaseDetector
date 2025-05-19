import sys
sys.modules['torch.classes'] = None # Exclude to prevent PyTorch/Streamlit compatibility error

import streamlit as st
import torch
import os
import matplotlib.pyplot as plt

from utils import load_model, preprocess_image, predict, get_device, attention_maps

device = get_device()

# Pytorch Model Prediction
def on_predict_click(image_file: str):
    # Load the pre-trained model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../swin_model_plant_disease_detector.pth"))
    model = load_model(model_path, device=device)
    
    # Preprocess the image
    image_tensor = preprocess_image(image_file)

    # Predict the class
    return predict(model=model, image_tensor=image_tensor, device=device)

def display_attention_map(image_file: str):
    """
    Display an attention map for Swin Transformer
    """
    st.header("Attention Maps:")

    # Preprocess image
    image_tensor = preprocess_image(image_file)
    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image = (image * 0.5 + 0.5).clip(0, 1)  

    fig, axes = plt.subplots(4, 3, figsize=(12, 16)) 
    axes = axes.flatten()

    for idx, attention_block in enumerate(attention_maps):
        ax = axes[idx]
        # reshape to remove the batch dimension
        attention_vector = attention_block[0].mean(0).mean(0)
        number_of_patches = attention_vector.shape[0]

        height = width = int(number_of_patches ** 0.5)

        # build the attention map
        attention_map = attention_vector.reshape(height, width).detach().cpu().numpy()
        attention_map = torch.nn.functional.interpolate(
            torch.tensor(attention_map)[None, None], size=image.shape[:2], mode='bilinear'
        )[0, 0].numpy()

        ax.imshow(image)
        ax.imshow(attention_map, alpha=0.5, cmap='jet')
        ax.set_title(f"Block {idx}", fontsize=8)
        ax.axis('off')

    st.pyplot(fig) 
    plt.close(fig)
    
def predict_page():
    st.header("Predict")
    
    # File uploader
    test_images = st.file_uploader("Upload image files (png, jpg, or jpeg) of plant leaves to predict their condition.", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if len(test_images) == 1:
        st.image(test_images[0], caption=test_images[0].name, use_container_width=True)

        if(st.button("Predict")):
            with st.spinner("Thinking..."):
                # Call the predict function
                prediction, confidence = on_predict_click(test_images[0])
                st.header("Prediction:")
                st.success(f"The model predicts that this is \"{prediction}\" with a confidence of {confidence*100:.2f}%.")
                display_attention_map(test_images[0])

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
                    st.image(test_image, caption=test_image.name, use_container_width=True)
                    st.success(f"The model predicts that the image \"{test_image.name}\" is \"{prediction}\" with a confidence of {confidence*100:.2f}%.")

    st.session_state.test_images = None
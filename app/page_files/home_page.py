import streamlit as st
import os

def home_page():
    st.header("Plant Disease Recognition System") 
    base_dir = os.path.abspath(os.path.join(__file__, "../../"))
    image_path = os.path.join(base_dir, "images", "home_page.jpeg")
    st.image(image_path, caption="Home Page", use_container_width=True)
    st.markdown(
        """
        This is a plant disease recognition system that helps farmers and agricultural professionals identify diseases in plants using machine learning techniques. 
        The system uses a pre-trained model to analyze images of plants and provide accurate predictions of potential diseases.
        """
    )   
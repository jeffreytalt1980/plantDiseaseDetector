import streamlit as st
import os

def home_page():
    st.header("Plant Disease Detection System") 
    base_dir = os.path.abspath(os.path.join(__file__, "../../"))
    image_path = os.path.join(base_dir, "images", "home_page.jpeg")
    st.image(image_path, caption="Home Page", use_container_width=True)
    st.markdown(
        """
        This is a plant disease recognition system. It can assist farmers in identification of diseases in plants. It uses a Swin Transformer
        trained using machine learning techniques. The model analyzes images of plants and provides predictions of potential diseases.
        """
    )   
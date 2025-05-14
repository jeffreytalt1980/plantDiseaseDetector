import streamlit as st

def home_page():
    st.header("Plant Disease Recognition System") 
    image_path = "images/home_page.jpeg"
    st.image(image_path, caption="Home Page", use_container_width=True)
    st.markdown(
        """
        This is a plant disease recognition system that helps farmers and agricultural professionals identify diseases in plants using machine learning techniques. 
        The system uses a pre-trained model to analyze images of plants and provide accurate predictions of potential diseases.
        """
    )   
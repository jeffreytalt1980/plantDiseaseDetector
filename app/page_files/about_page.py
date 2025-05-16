import streamlit as st

def about_page():
    st.header("About This Project")
    st.markdown(
        """
        ### About the Model
        This uses used a pretrained Swin Transformer from the timm library that was fine-tuned on the PlantVillage dataset.
        It utilized the 'swin_tiny_patch4_window7_224' architecture which has patch size of 4, window size of 7, and an input image size of 224x224.
        The model was trained for 10 epochs with a batch size of 32, and a learning rate of 0.0001.
        Versions of it were trained on a NVIDIA GeForce GTX 1650 GPU with 4GB of VRAM and TPU at Google Colab. 

        ### About the Dataset
        This version of the PlantVillage dataset is was downloaded from TensorFlow Datasets. 
        It can be found at: https://www.tensorflow.org/datasets/catalog/plant_village
       
        #### Content: 54,305 images across 38 classes
        - Training: 70%
        - Validation: 20%
        - Testing: 10%
       
        #### More Information
        """
    )

    more_info = st.expander("Details...")
    with more_info: 
        st.markdown(
            """
                - **Apple: Scab**: 63 samples
                - **Apple: Black Rot**: 63 samples
                - **Apple: Cedar Apple Rust**: 28 samples
                - **Apple: Healthy**: 165 samples
                - **Blueberry: Healthy**: 151 samples
                - **Cherry: Powdery Mildew**: 106 samples
                - **Cherry: Healthy**: 87 samples
                - **Corn: Cercospora Leaf Spot/Gray Leaf Spot**: 52 samples
                - **Corn: Common Rust**: 120 samples
                - **Corn: Northern Leaf Blight**: 99 samples
                - **Corn: Healthy**: 117 samples
                - **Grape: Black Rot**: 118 samples
                - **Grape: Esca (Black Measles)**: 139 samples
                - **Grape: Leaf Blight (Isariopsis Leaf Spot)**: 108 samples
                - **Grape: Healthy**: 43 samples
                - **Orange: Huanglongbing (Citrus Greening)**: 552 samples
                - **Peach: Bacterial Spot**: 231 samples
                - **Peach: Healthy**: 37 samples
                - **Bell Pepper: Bacterial Spot**: 101 samples
                - **Bell Pepper: Healthy**: 149 samples
                - **Potato: Early Blight**: 100 samples
                - **Potato: Late Blight**: 100 samples
                - **Potato: Healthy**: 16 samples
                - **Raspberry: Healthy**: 38 samples
                - **Soybean: Healthy**: 509 samples
                - **Squash: Powdery Mildew**: 184 samples
                - **Strawberry: Leaf Scorch**: 112 samples
                - **Strawberry: Healthy**: 46 samples
                - **Tomato: Bacterial Spot**: 214 samples
                - **Tomato: Early Blight**: 100 samples
                - **Tomato: Late Blight**: 192 samples
                - **Tomato: Leaf Mold**: 96 samples
                - **Tomato: Septoria Leaf Spot**: 178 samples
                - **Tomato: Spider Mites (Two-spotted Spider Mite)**: 168 samples
                - **Tomato: Target Spot**: 142 samples
                - **Tomato: Yellow Leaf Curl Virus**: 537 samples
                - **Tomato: Mosaic Virus**: 38 samples
                - **Tomato: Healthy**: 160 samples
            """
        )
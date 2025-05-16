import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def accuracy_visualization_page():
    """
    This function is used to display a page to visualize the accuracy of the model over training epochs.
    """

    history_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../history.json"))

    st.header("Accuracy Visualization")
    st.markdown(
        """
        This is a visualization of the accuracy of the model during it's training.
        """
    )

    with st.spinner("Loading..."): 
        with open(history_path, 'r') as f:
            history = json.load(f)

        training_accuracy = history["training_accuracy"]
        validation_accuracy = history["validation_accuracy"]

        plt.figure(figsize=(8, 5))
        plt.plot(training_accuracy, label="Training Accuracy", marker='o')
        plt.plot(validation_accuracy, label="Validation Accuracy", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Across Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        st.pyplot(plt)
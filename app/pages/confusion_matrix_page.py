import streamlit as st
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from constants import total_labels, total_predictions

def confusion_matrix_page():
    st.header("Confusion Matrix")
    st.markdown(
        """
        This is confusion matrix that results from the data recorded from the evaluating against test set
        of this model.
        """
    )
    
    with st.spinner("Loading..."):
        matrix = confusion_matrix(total_labels, total_predictions)

        plt.figure(figsize=(40,40))
        sns.heatmap(matrix, annot=True, annot_kws={'size':10})
        plt.xlabel("Predicted Class", fontsize=20)
        plt.ylabel("Actual Class", fontsize=20)
        plt.title("Plant Disease Prediction Confusion Matrix", fontsize=30)
        
        st.pyplot(plt)

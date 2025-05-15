import streamlit as st

from page_files.home_page import home_page
from page_files.about_page import about_page
from page_files.predict_page import predict_page
from page_files.confusion_matrix_page import confusion_matrix_page
from page_files.classification_report_page import classification_report_page

def navigation_switch(app_mode: str):
    """
    Switch between different app modes
    """
    match app_mode:
        case "Home":
            home_page()
        case "About":
            about_page()
        case "Predict":
            predict_page()
        case "Confusion Matrix":
            confusion_matrix_page()
        case "Classification Report":
            classification_report_page()
        case _:
            home_page()

# Sidebar
st.sidebar.title("Navigation")    
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Predict", "Confusion Matrix", "Classification Report"])

navigation_switch(app_mode)


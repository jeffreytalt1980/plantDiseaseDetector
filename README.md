## About
This is Jeffrey Alt's Plant Disease Prediction Project for the Troy University course Specialized Study in Computer Science - CS6625.

Utilized a Swin Tranformer model trained with PyTorch to predict the classification of disease/healthy status when presented with an image of a leaf.

## Pages

### Home
Splash Page

### Predict
Prediction Tool. Accepts single files and multiple. Will immediately display the image if a single file is uploaded. If multiple files are uploaded, they will be displayed with their resultant prediction. Prediction includes a cleaned up version of the class name and the confidence.

### Classification Report 
Displays the classification report for the model.

### Confusion Matrix
Displays the confusion matrix for the model.

### Accuracy Visualization
Displays the recorded accuracy for training and test sets during the course of the epochs of the training loop.

### About
Information about the model and dataset

## Built With
* Python 
* Streamlit
* Jupyter Notebook
* VSCode
* Google Colab
* CUDA
* Git
* Git-lfs

### Dependencies:
* PyTorch
* TorchVision
* TIMM
* Scikit Learn
* Pillow
* Pandas
* NumPy
* PyArrow
* MatPlotLib
* Seaborn

## Installation Instructions
1. Build the docker file:
```
sudo docker build -t plant-disease-detector-image:v1.0
```
2. Run the dockerfile:
```
sudo docker run -p 8501:8501 plant-disease-detector-image:v1.0
```
3. Navigate to the website: http://localhost:8501

## Link to share.streamlit.io Deployment
The application is also deployed on share.streamlit.io:
[Plant Disease Detection System](https://plantdiseasedetector-jeffreytalt.streamlit.app/)
## About
This is Jeffrey Alt's Plant Disease Prediction Project for the Troy University course Specialized Study in Computer Science - CS6625.

This application is used to detect the status of leaves within 38 combinations of species/disease (or healthy status). It can be used by farmers or agricultural professionals to get ahead of diseases to treat or isolate diseased specimens and in turn prevent loss of crops. This important to preventing food insecurity as the world population continues to grow and various natural and man-made factors impact the stability of crop yields. 

### Model Summary
This application utilizes a Swin Tranformer model trained with PyTorch to predict the classification of disease/healthy status when presented with an image of a leaf. It was trained from a timm 'swin_tiny_patch4_window7_224' architecture using Google Colab resources and following the steps defined in the included Jupyter notebook. A CNN model was also trained as a baseline (not included). The model version used in the web application was produced during my second to last run and included extensive data augmentation techniques, label smoothing, dropout, and an optimizer running the AdamW algorithm. The final run which included weighted decay produced a product with inferior confidence scores for images that did not contain leaves. 

### Dataset
This model uses the a version of the PlantVillage dataset downloaded from TensorFlow's Datasets Catalog. It contains 53,305 samples across 38 classes. It is as close to the original, unaugmented PlantVillage dataset that was readily available. The dataset can be found at: [Plant Village](https://www.tensorflow.org/datasets/catalog/plant_village). All pre-processing occurs within the training pipline defined in the included Jupyter notebook. The dataset is included in the repository in the /plantDiseaseDetector/PlantVillage directory.

### Key Results
Metrics for the model can be found in the Jupyter notebook included with this application as well as in the application on the classification report page.

## Pages

### Home
This is the splash page for the application.

### Predict
Prediction Tool. Accepts single files and multiple. Will immediately display the image if a single file is uploaded. If multiple files are uploaded, they will be displayed with their resultant prediction and the series of attention maps that were generated from the transformers attention weights. Prediction includes a cleaned up version of the class name and the confidence.

### Classification Report 
Displays the classification report for the model.

### Confusion Matrix
Displays the confusion matrix for the model.

### Accuracy Visualization
Displays the recorded accuracy for training and test sets during the course of the epochs of the training loop.

### About the Application
Displays information about the model and dataset.

## Built With
* Python 
* Streamlit
* Jupyter Notebook
* VSCode
* Google Colab
* CUDA
* Git
* Git-lfs
* Docker

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
After downloading the code from GitHub, navigate to the root folder of the project (/plantDiseaseDetector). Then choose one of the following ways to run the software.

### With Docker
1. Build the docker file:
```
sudo docker build -t plant-disease-detector-image:v1.0 .
```
2. Run the dockerfile:
```
sudo docker run -p 8501:8501 plant-disease-detector-image:v1.0
```
3. Navigate to the website: http://localhost:8501

### On Local Machine
1. Install the dependencies:
```
pip install requirements.txt
```
2. Start Streamlit Application:
```
streamlit run app/app.py
```
3. Navigate to the website: http://localhost:8501

## Usage Instructions
To conduct a prediction, upload an image (or images) file in jpg or png format. If you upload a singular image, it will be immediately shown with a "Predict" button. Clicking the button will return a prediction with associated confidence value for the image. If you upload a group of images, a "Predictions" button will be displayed. Clicking the button will print each image with its associated prediction and confidence score.

## Link to share.streamlit.io Deployment
The application is also deployed on share.streamlit.io:
[Plant Disease Detection System](https://plantdiseasedetector-jeffreytalt.streamlit.app/).

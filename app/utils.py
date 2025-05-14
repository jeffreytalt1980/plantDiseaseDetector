import torch 
import timm
from torchvision import transforms
from PIL import Image

from constants import IMAGE_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD, class_names, class_names_output_map

def load_model(path: str, device: torch.device) -> torch.nn.Module:
    """
    Load the pre-trained model
    """
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=len(class_names))
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)
    return model
    
def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Preprocess the image for model prediction
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)  
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def get_prediction_label(prediction: int) -> str:   
    """
    Get the prediction label from the model output
    """
    return class_names_output_map[class_names[prediction]]

def predict(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> str:
    """
    Make a prediction using the model
    """
    image_tensor = image_tensor.to(device)

    # Make prediction
    with torch.no_grad():  
        prediction_weights = model(image_tensor)
    
    prediction = torch.argmax(prediction_weights, dim=1).item()
    confidence = torch.softmax(prediction_weights, dim=1).max().item()

    print(torch.softmax(prediction_weights, dim=1))

    return get_prediction_label(prediction), confidence


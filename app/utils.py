import sys
sys.modules['torch.classes'] = None # Exclude to prevent PyTorch/Streamlit compatibility error

import torch 
import timm
from torchvision import transforms
from PIL import Image

from constants import IMAGE_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD, class_names, class_names_output_map
from model.CustomHead import CustomHead

attention_maps = []

def load_model(path: str, device: torch.device) -> torch.nn.Module:
    """
    Load the pre-trained model
    """
    number_of_classes = len(class_names)
    # create the model
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=number_of_classes)

    # replace with custom head
    in_features = model.head.in_features
    model.head = CustomHead(in_features, number_of_classes)
  
    # replace the weights with weights saved from training
    model.load_state_dict(torch.load(path, map_location=device))
    
    model.eval()
    model.to(device)

    # Register the attention hook for each block
    for layer in model.layers:
        for block in layer.blocks:
            block.attn.attn_drop.register_forward_hook(get_attention_hook)
    
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
    attention_maps.clear()
    image_tensor = image_tensor.to(device)

    # Make prediction
    with torch.no_grad():  
        prediction_weights = model(image_tensor).mean(dim=1)

    prediction = torch.argmax(prediction_weights, dim=1).item()
    confidence = torch.softmax(prediction_weights, dim=1).max().item()

    return get_prediction_label(prediction), confidence

def get_attention_hook(module, input, output):
    """
    Hook to get the attention weights
    """
    return attention_maps.append(output)

def get_device() -> torch.device:
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
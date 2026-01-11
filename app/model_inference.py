import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
from utils import get_pretrained_model
from os.path import join

def predict_vit(image: Image.Image) -> dict:
    """
    Get the class prediction and prediction confidence for an input image using a pretrained vit model

    Args:
        image (Image.Image): The input image

    Returns:
        dict: A dictionary containing the predicted class and the confidence prediction
    """
    
    # instatiate the vit-16-224 processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # instatiate the vit-16-224 model
    # this is finetuned on imagenet1k which contains the cat and dog class (along with specific brieds of dogs)
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # process the input image
    inputs = processor(images=image, return_tensors="pt")
    
    # get the model output prediction logits
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    
    # get the class prediction and the confidence
    predicted_class_idx = logits.argmax(-1).item()
    confidence = torch.softmax(logits, dim=-1).max().item()
    
    # map the prediction class id to a string class
    label = model.config.id2label[predicted_class_idx]
    
    # format the output
    if 'cat' in label.lower():
        display_label = "Cat"
    elif 'dog' in label.lower():
        display_label = "Dog"
    else:
        display_label = f"Other ({label})"
        
    return {"class_prediction": display_label, "confidence": confidence}

def predict_finetuned(image: Image.Image,
                      model_name: str) -> dict:
    """
        Get the class prediction and prediction confidence for an input image using a finetuned model


    Args:
        image (Image.Image): The input image
        model_name (str): The model name

    Returns:
        dict: A dictionary containing the predicted class and the confidence prediction
    """
    
    
    # Load Architecture
    model = get_pretrained_model(model_name.lower(),
                                 num_classes=2)
    
    # load model weights
    try:
        model.load_state_dict(torch.load(join(".", "checkpoints", f"{model_name}_cat_dog_model.pth"),
                                         map_location='cpu'))
    except FileNotFoundError:
        return "Error: Model not trained yet", 0.0

    # set model to evaluation mode
    model.eval()
    
    
    # apply transformations
    dtransform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    transformed_image = dtransform(image).unsqueeze(0)
    
    
    # get model outputs and the prediction confidence
    with torch.no_grad():
        outputs = model(transformed_image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
    classes = ['Cat', 'Dog']
    return {"class_prediction": classes[predicted.item()],
            "confidence": confidence.item()}

if __name__ == "__main__":
    from PIL import Image
    
    # path = "./stuctured_dataset2/test/Cat/1.jpg"
    path = "./stuctured_dataset2/test/Dog/12515.jpg"
    
    # vit
    print("pretrained")
    res_pretrained = predict_vit(Image.open(path))
    print(res_pretrained)
    
    # finetuned
    print("finetuned")
    res_finetuned = predict_finetuned(Image.open(path))
    print(res_finetuned)
    
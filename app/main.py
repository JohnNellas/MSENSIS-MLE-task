from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
from app.model_inference import predict_vit, predict_finetuned

app = FastAPI(title="Cat/Dog Classifier")

@app.get("/")
def read_root() -> dict:
    """API endpoint for roow

    Returns:
        dict: a dictionary containing a simple message for the root api endpoint
    """
    return {"message": "Welcome to the API for cat and dog image classification"}

@app.post("/predict")
async def predict(
    model_type: str = Form(...),
    file: UploadFile = File(...)
) -> dict:
    """The api endpoint for the classification of the image provided by the user

    Args:
        model_type (str, optional): The model to be utilized for prediction. Defaults to Form(...).
        file (UploadFile, optional): the user provided image . Defaults to File(...).

    Returns:
        dict: the prediction, its confidence and the employed model.
    """
    # read the image provided by the user
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Route to the desired model and return the class prediction, the confidence and the model type
    if model_type == "Hugging Face ViT":
        res = predict_vit(image)
    else:
        res = predict_finetuned(image, model_type)
    
    
    label, confidence = res["class_prediction"], res["confidence"] 
    
    return {
        "prediction": label,
        "confidence": float(confidence),
        "model_used": model_type
    }
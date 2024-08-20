from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles  # Import StaticFiles
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os

app = FastAPI()

# Load the pre-trained model
model = tf.keras.models.load_model("dogs_cats.h5")

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Mount the static directory for serving CSS and images
app.mount("/static", StaticFiles(directory="static"), name="static")

def preprocess_image(image: Image.Image) -> np.array:
    image = image.resize((150, 150))  # Resize to the model input size
    image = np.array(image) / 255.0   # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image: np.array) -> str:
    predictions = model.predict(image)
    y_pred = 'Dog' if predictions[0][0] >= 0.5 else 'Cat'
    return y_pred

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-file", response_class=HTMLResponse)
async def predict_file(request: Request, file: UploadFile = File(...)):
    # Read and save the image temporarily
    image = Image.open(BytesIO(await file.read()))
    image_path = "static/uploaded_image.png"
    image.save(image_path)
    
    processed_image = preprocess_image(image)
    prediction = predict(processed_image)
    
    return templates.TemplateResponse("prediction.html", {
        "request": request,
        "prediction": prediction,
        "image_url": f"/{image_path}"
    })

@app.post("/predict-url", response_class=HTMLResponse)
async def predict_url(request: Request, url: str = Form(...)):
    # Fetch the image from the URL
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image_path = "static/uploaded_image.png"
    image.save(image_path)

    processed_image = preprocess_image(image)
    prediction = predict(processed_image)
    
    return templates.TemplateResponse("prediction.html", {
        "request": request,
        "prediction": prediction,
        "image_url": f"/{image_path}"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

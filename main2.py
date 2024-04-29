import os
from PIL import Image
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

# Load the trained model


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]

)

MODEL = tf.keras.models.load_model('myOwn.keras')

# Define the index route
@app.get('/')
async def index():
    return {"Message": "This is Index"}

# Define the prediction route
@app.post('/predict/')
async def predict(file: UploadFile = File(...)):

    # Read the image file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    # Convert image to RGB format (discard extra channels)
    img = img.convert('RGB')
    
    # Preprocess the image
    img = img.resize((120, 120))  # Resize the image to match the model's input shape
    img = np.array(img) / 255.0   # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Print out the shape of the input tensor
    print("Input tensor shape:", img.shape)
    
    
    # Make prediction
    prediction = MODEL.predict(img)
    print(prediction)
    percentage = round(prediction[0][0] * 100, 2)
    predicted_class = "Pneumonia" if prediction > 0.7 else ("Risk" if 0.4 <= prediction <= 0.6 else "Normal")

    return {"prediction": predicted_class, "percentage": percentage}

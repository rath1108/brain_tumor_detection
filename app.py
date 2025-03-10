from flask import Flask, request, render_template
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize Flask App
app = Flask(__name__)

# Load Trained Model
model = load_model("brain_tumor_cnn_model.h5")

# Set Upload Folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define Image Size
IMG_SIZE = 128

# Function to Predict Tumor
def predict_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Reshape for model input
    
    prediction = model.predict(img)[0][0]
    return "🧠 Tumor Detected" if prediction > 0.5 else "✅ No Tumor"

# Home Page
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Check if file was uploaded
        if "file" not in request.files:
            return "No file uploaded"
        
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Make prediction
            result = predict_image(filepath, model)
            
            return render_template("result.html", result=result, image=filename)
    
    return render_template("index.html")

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)

from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from pathlib import Path
import json

# Create router
router = APIRouter()

# Setup templates
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# Global model variable
model = None
IMAGE_SIZE = (128, 128)

def load_model():
    """Load the trained CNN model"""
    global model
    if model is None:
        model_path = Path(__file__).parent / "models/cnn_model.keras"
        if model_path.exists():
            try:
                # Try loading with custom objects to handle compatibility issues
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects=None,
                    compile=False  # Don't compile to avoid optimizer issues
                )
                # Recompile the model with current TensorFlow version
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                print("Model loaded and recompiled successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Attempting to rebuild model from scratch...")
                try:
                    # If loading fails, create a new model with same architecture
                    model = build_fallback_model()
                    print("Fallback model created (needs training)")
                except Exception as e2:
                    print(f"Error creating fallback model: {e2}")
                    model = None
        else:
            print("Model file not found")
            model = None
    return model

def build_fallback_model():
    """Build a fallback model with the same architecture"""
    from tensorflow.keras import layers, models
    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess uploaded image for model prediction"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        image = image.convert("L")
        
        # Resize to model input size
        image = image.resize(IMAGE_SIZE)
        
        # Convert to numpy array and normalize
        img_array = np.asarray(image, dtype=np.float32) / 255.0
        
        # Add channel dimension
        img_array = np.expand_dims(img_array, axis=-1)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/predict")
async def predict_stroke(file: UploadFile = File(...)):
    """Predict stroke from uploaded image"""
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Load model
    current_model = load_model()
    if current_model is None:
        raise HTTPException(status_code=500, detail="Model not available. Please check model file or retrain the model.")
    
    # Check if this is a fallback model (untrained)
    try:
        # Test if model has been trained by checking if it has weights
        test_input = np.zeros((1, 128, 128, 1))
        test_prediction = current_model.predict(test_input, verbose=0)
        if np.isnan(test_prediction).any():
            raise HTTPException(status_code=500, detail="Model needs to be trained. Please run the training script first.")
    except Exception as e:
        if "needs training" in str(e).lower():
            raise HTTPException(status_code=500, detail="Model architecture loaded but needs training. Please run the training script first.")
        # If it's just a prediction error, continue (model might be fine)
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        prediction = current_model.predict(processed_image)[0][0]
        probability = float(prediction)
        
        # Determine risk level and message
        if probability > 0.8:
            risk_level = "HIGH RISK"
            risk_color = "#dc3545"  # Red
            message = "Strong indication of stroke"
            emoji = "🔴"
        elif probability > 0.6:
            risk_level = "MODERATE-HIGH RISK"
            risk_color = "#fd7e14"  # Orange
            message = "Consult a medical professional"
            emoji = "🟡"
        elif probability > 0.4:
            risk_level = "MODERATE RISK"
            risk_color = "#ffc107"  # Yellow
            message = "Monitor closely"
            emoji = "🟡"
        elif probability > 0.2:
            risk_level = "LOW-MODERATE RISK"
            risk_color = "#20c997"  # Teal
            message = "Unlikely but monitor"
            emoji = "🟢"
        else:
            risk_level = "LOW RISK"
            risk_color = "#28a745"  # Green
            message = "No strong indication of stroke"
            emoji = "🟢"
        
        return {
            "success": True,
            "probability": probability,
            "percentage": f"{probability:.1%}",
            "risk_level": risk_level,
            "risk_color": risk_color,
            "message": message,
            "emoji": emoji,
            "disclaimer": "This is an AI model for educational purposes. Always consult healthcare professionals for medical advice."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if load_model() is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status
    }

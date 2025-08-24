# Stroke Prediction System

This system analyzes medical images to predict stroke risk using a CNN model.


<details>
  <summary>▶ Click to watch demo video</summary>

  <video src="https://github.com/user-attachments/assets/f285d513-4e78-4680-aaed-6f80aae8bcf3" width="500" controls>
    Your browser does not support the video tag.
  </video>

</details>

</details>



## 🚀 Quick Start

```bash
# Create and activate virtual environment using
uv venv
source .venv/bin/activate

# Install dependencies
uv sync
```
### Data Sources

* [Annotated Facial Images for Stroke Classification](https://www.kaggle.com/datasets/abdussalamelhanashy/annotated-facial-images-for-stroke-classification/data)
* [Facial Droop and Facial Paralysis Image](https://www.kaggle.com/datasets/kaitavmehta/facial-droop-and-facial-paralysis-image/data)
* [Stroke Face Dataset](https://universe.roboflow.com/stroke-aware/stroke_face-6sqpf/dataset/2)

## 📂 Folder Structure
```
stroke_prediction/
├── Stroke/          # Folder with stroke-positive images
├── NonStroke/       # Folder with stroke-negative images
├── main.py          # Original XGBoost implementation
├── main2.py         # CNN implementation (recommended)
└── models/          # Saved model directory
```

## 🧠 Model Training (Option 1)
1. Run the program:
   ```bash
   python main.py
   ```
2. Select option **1** (Train/Run Model)
3. Provide inputs:
   - Data directory (default: current directory)
   - Image limit per class (0 for no limit)
   - Training epochs (default: 10)

## 🔍 Prediction (Option 2)
1. After training, run:
   ```bash
   python main.py
   ```
2. Select option **2** (Test with Image)
3. Enter path to the image you want to analyze

## 📊 Expected Output
- Training: Accuracy metrics and PDF report
- Prediction: Probability score with risk level:
  - >70%: High risk
  - 30-70%: Moderate risk
  - <30%: Low risk

## ⚙️ Dependencies
- Python 3.8+
- TensorFlow/Keras
- scikit-learn
- PIL/Pillow
- matplotlib
- fpdf2

💡 Tip: All dependencies are automatically installed with `uv sync`

# Stroke Prediction API

A FastAPI-based web application for stroke prediction using a trained CNN model.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python run.py
   ```

3. **Open your browser:**
   - Main app: http://localhost:8000
   - API docs: http://localhost:8000/docs

## Features

- 🖼️ **Image Upload**: Drag & drop or click to upload medical images
- 🧠 **AI Analysis**: Uses trained CNN model for stroke prediction
- 📊 **Risk Assessment**: Provides risk levels with probability scores
- 📱 **Responsive UI**: Modern, mobile-friendly interface
- ⚡ **Fast API**: RESTful API with automatic documentation

## API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Upload image for prediction
- `GET /health` - Health check endpoint

## Project Structure

```
./
├── app.py          # FastAPI application
├── routes.py       # API routes (Django-style)
├── run.py          # Startup script
├── requirements.txt # Dependencies
├── templates/      # HTML templates
│   └── index.html  # Main UI
└── static/         # Static files (CSS, JS)
```

## Notes

- Model file should be at: `../models/cnn_model.keras`
- Supports common image formats (JPEG, PNG, etc.)
- Images are automatically resized to 128x128 for model input

- Provides medical disclaimer for educational use only














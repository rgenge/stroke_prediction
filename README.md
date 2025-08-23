# Stroke Prediction System

This system analyzes medical images to predict stroke risk using a CNN model.

## 🚀 Quick Start

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv sync
```
Copy data and put in two folders Non Stroke and Stroke :
https://www.kaggle.com/datasets/abdussalamelhanashy/annotated-facial-images-for-stroke-classification/data
https://www.kaggle.com/datasets/kaitavmehta/facial-droop-and-facial-paralysis-image/data

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

## 📂 Folder Structure
```
stroke_prediction/
├── Stroke/          # Folder with stroke-positive images
├── NonStroke/       # Folder with stroke-negative images
├── main.py          # Original XGBoost implementation
├── main2.py         # CNN implementation (recommended)
└── models/          # Saved model directory
```

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

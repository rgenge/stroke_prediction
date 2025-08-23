#!/usr/bin/env python3
"""
Startup script for the Stroke Prediction FastAPI application
"""
import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Import the app so it's available for uvicorn
from app import app

# Now import and run the app
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Stroke Prediction API...")
    print("📍 Open your browser to: http://localhost:8000")
    print("📋 API docs available at: http://localhost:8000/docs")
    print()
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

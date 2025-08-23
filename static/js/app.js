// Stroke Prediction App JavaScript

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const predictBtn = document.getElementById('predictBtn');
const loading = document.getElementById('loading');
const result = document.getElementById('result');
const error = document.getElementById('error');

let selectedFile = null;

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
});

function initializeEventListeners() {
    // Drag and drop functionality
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // File input change
    fileInput.addEventListener('change', handleFileInputChange);
    
    // Predict button click
    predictBtn.addEventListener('click', handlePredict);
}

// Drag and Drop Handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave() {
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
}

// File Input Handler
function handleFileInputChange(e) {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
}

// File Selection Handler
function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file.');
        return;
    }

    selectedFile = file;
    
    // Show image preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'block';
        predictBtn.style.display = 'inline-block';
        hideError();
        hideResult();
    };
    reader.readAsDataURL(file);
}

// Prediction Handler
async function handlePredict() {
    if (!selectedFile) {
        showError('Please select an image first.');
        return;
    }

    showLoading();
    hideError();
    hideResult();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            showResult(data);
        } else {
            showError(data.detail || 'Prediction failed');
        }
    } catch (err) {
        console.error('Prediction error:', err);
        showError('Network error. Please try again.');
    } finally {
        hideLoading();
    }
}

// UI State Management Functions
function showLoading() {
    loading.style.display = 'block';
    predictBtn.disabled = true;
}

function hideLoading() {
    loading.style.display = 'none';
    predictBtn.disabled = false;
}

function showResult(data) {
    document.getElementById('resultEmoji').textContent = data.emoji;
    document.getElementById('resultRisk').textContent = data.risk_level;
    document.getElementById('resultRisk').style.color = data.risk_color;
    document.getElementById('resultProbability').textContent = data.percentage;
    document.getElementById('resultProbability').style.color = data.risk_color;
    document.getElementById('resultMessage').textContent = data.message;
    result.style.display = 'block';
    result.style.background = data.risk_color + '10';
    result.style.border = '1px solid ' + data.risk_color + '40';
}

function hideResult() {
    result.style.display = 'none';
}

function showError(message) {
    error.textContent = message;
    error.style.display = 'block';
}

function hideError() {
    error.style.display = 'none';
}

// Utility Functions
function openFileDialog() {
    fileInput.click();
}

// Export functions for global access (if needed)
window.openFileDialog = openFileDialog;

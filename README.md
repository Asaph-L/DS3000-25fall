## Project Overview

This project is a deep learning pipeline for plant disease classification using leaf images. It includes data cleaning, model training (ResNet50, EfficientNet-B0, CNN-20), evaluation, and report generation. The workflow is designed for reproducible experiments and comprehensive analysis, including training history, cross-validation, and visualizations.

## Setup Instructions

### 1. Requirements

See `requirements.txt` for all dependencies.

### 2. Installation

#### Windows
1. Install [Python 3.9+](https://www.python.org/downloads/).
2. Open PowerShell and navigate to the project folder.
3. Create a virtual environment:
	```powershell
	python -m venv venv
	.\venv\Scripts\Activate.ps1
	```
4. Install requirements:
	```powershell
	pip install -r requirements.txt
	```

#### Linux/macOS
1. Install Python 3.9+ (e.g., `sudo apt install python3 python3-venv python3-pip` on Ubuntu).
2. Open a terminal and navigate to the project folder.
3. Create a virtual environment:
	```bash
	python3 -m venv venv
	source venv/bin/activate
	```
4. Install requirements:
	```bash
	pip install -r requirements.txt
	```

## Project Structure

```
DS3000-25fall/
├── README.md                # Project overview and setup instructions
├── requirements.txt         # Python dependencies
├── ModelTraining.ipynb      # Jupyter notebook for model training
├── ModelInference.ipynb     # Jupyter notebook for model evaluation/inference
├── ModelTraning_Output_*.html # Exported HTML reports from notebooks
├── ModelInference_output_*.html # Exported HTML reports from inference
├── data/                    # Data directory
│   ├── archive/             # Original dataset (unmodified)
│   │   └── Dataset/         # Raw image folders by class
│   └── clean/               # Cleaned dataset for training/testing
├── models/                  # Saved model weights and checkpoints
│   ├── best_cnn_20.pth      # Best CNN-20 model (single best, all data)
│   ├── best_efficientnet_b0.pth  # Best EfficientNet-B0 model (single best, all  
|   |                               data)
│   ├── best_resnet50.pth   # Best ResNet50 model (single best, all data)
│   ├── best_cnn20_foldx.pth # The best CNN-20 model in x-fold CV
|   ├── best_efficientnet_foldx.pth # The best EfficientNet model in x-fold CV
|   ├── best_resnet50_foldx.pth # The best ResNet50 model in x-fold CV
|   ├── cnn_20_complete.pth    # Custom CNN-20 model with full training history  
|   |                            and metadata  
|   ├── efficientnet_b0_complete.pth  # EfficientNet-B0 model with full training  
|   |                                   history and metadata  
├── └── resnet50_complete.pth        # ResNet50 model with full training history  
|                                      and metadata
└── ...                      # Other files (scripts, outputs, etc.)
```

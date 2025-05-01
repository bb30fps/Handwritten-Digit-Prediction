# Handwritten Digit Recognition System

A complete implementation of a Convolutional Neural Network (CNN) for MNIST digit classification with a graphical user interface for real-time predictions. Built with TensorFlow and Tkinter.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [System Requirements](#-system-requirements)
- [Installation Guide](#-installation-guide)
- [Usage Instructions](#-usage-instructions)
- [Training Process & Logging](#-training-process--logging)
- [GUI Operation](#-gui-operation)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🌟 Project Overview

### Model Architecture
- **CNN Structure**:
  - 2 Conv2D layers (32 and 64 filters)
  - 2 MaxPooling layers
  - 2 Dense layers (128 and 64 units) with Dropout
  - Adam optimizer (0.001 learning rate)
  - Categorical crossentropy loss
- **Performance**: Achieves ~99% test accuracy on MNIST dataset

### GUI Features
- Interactive canvas for digit drawing
- Real-time prediction with confidence score
- Image preprocessing pipeline (28x28 grayscale conversion)
- Clear canvas functionality

---

## 💻 System Requirements
- Python 3.8+
- 4GB RAM minimum
- 500MB Disk space
- pip package manager
- (Optional) NVIDIA GPU for accelerated training

---

## 📥 Installation Guide

### 1. Clone Repository
```bash
git clone https://github.com/bb30fps/handwritten-digit-recognizer.git
cd handwritten-digit-recognizer
2. Create Virtual Environment (Recommended)
bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
3. Install Dependencies
bash
pip install -r requirements.txt
🚀 Usage Instructions
Model Training
bash
python train.py \
  --epochs 15 \
  --batch_size 128 \
  --model_dir models \
  --log_dir logs
Arguments:

--epochs: Number of training iterations (default: 15)

--batch_size: Samples per gradient update (default: 128)

--model_dir: Output directory for model checkpoints

--log_dir: Directory for training logs and metrics

Output Files:

models/my_model.keras: Best model during training

models/model.keras: Final trained model

logs/training_metrics.png: Accuracy/loss visualization

logs/logs/: TensorBoard event files

GUI Application
bash
python gui.py
Interface Components:

Drawing Canvas (300x300 pixels)

Recognition Result Display

"Recognize" Button: Trigger prediction

"Clear" Button: Reset canvas

📊 Training Process & Logging
Callback System
Model Checkpointing:

Saves best model based on validation accuracy

Location: models/my_model.keras

Early Stopping:

Monitors validation loss

Stops training if no improvement for 3 epochs

TensorBoard Integration:

Track metrics in real-time:

bash
tensorboard --logdir logs/logs
Access at http://localhost:6006

Training Visualization
Training Metrics

🖥️ GUI Operation
Workflow
Draw digit using mouse/touchpad

Click "Recognize"

System displays predicted digit and confidence

Click "Clear" to reset

Image Preprocessing Pipeline
Capture canvas area

Convert to grayscale

Resize to 28x28 pixels

Invert colors (MNIST-compatible format)

Normalize pixel values [0, 1]

📂 Project Structure
.
├── models/               # Saved models
│   ├── model.keras      # Final trained model
│   └── my_model.keras   # Best validation model
├── logs/                # Training artifacts
│   ├── logs/            # TensorBoard logs
│   └── training_metrics.png
├── gui.py               # GUI application
├── train.py             # Model training script
├── requirements.txt     # Dependency list
└── README.md

🛠 Troubleshooting
Issue: Model not found at models/model.keras
Solution: Run train.py first to generate model files

Issue: Dependency conflicts
Solution: Use virtual environment and ensure correct Python version

Issue: Low prediction accuracy
Solution:

Draw centered, clear digits

Ensure proper inversion in preprocessing

Retrain with more epochs

Issue: TensorFlow GPU errors
Solution: Install CPU-only version:

bash
pip uninstall tensorflow
pip install tensorflow-cpu

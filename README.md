# Handwritten Digit Recognition System

A complete implementation of a Convolutional Neural Network (CNN) for MNIST digit classification with a graphical user interface for real-time predictions. Built with TensorFlow and Tkinter.

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

## 📥 Installation Guide

### 1. Clone Repository

git clone https://github.com/bb30fps/handwritten-digit-recognizer.git
cd handwritten-digit-recognizer

2. Create Virtual Environment (Recommended)

python -m venv venv
venv\Scripts\activate

3. Install Dependencies

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

GUI Application:

python gui.py

Interface Components:

Drawing Canvas (300x300 pixels)
Recognition Result Display
"Recognize" Button: Trigger prediction
"Clear" Button: Reset canvas

📊 Logging & Model Saving

Automatic Logging: During training, logs are generated via TensorBoard and stored in logs/logs/. 

These include:

-Training/validation accuracy and loss.
-Histograms of layer weights.
-Model Checkpoints: The best model (based on validation accuracy) is saved to models/my_model.keras.
-Final Model: The trained model is saved as models/model.keras after training completes.
-Training Metrics: A plot of accuracy and loss curves is saved to logs/training_metrics.png.

TensorBoard Integration:

-Track metrics in real-time
-tensorboard --logdir logs/logs
Access at http://localhost:6006

🖥️ GUI Operation:

Workflow:

-Draw digit using mouse/touchpad
-Click "Recognize"
-System displays predicted digit and confidence
-Click "Clear" to reset
-Image Preprocessing Pipeline
-Capture canvas area
-Convert to grayscale
-Resize to 28x28 pixels
-Invert colors (MNIST-compatible format)
-Normalize pixel values [0, 1]

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
-Draw centered, clear digits
-Ensure proper inversion in preprocessing
-Retrain with more epochs

Issue: TensorFlow GPU errors
Solution: Install CPU-only version:

pip uninstall tensorflow
pip install tensorflow-cpu

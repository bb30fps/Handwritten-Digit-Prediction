# Handwritten Digit Recognition System

A GUI-based application that recognizes handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Built with TensorFlow for the machine learning model and Tkinter for the graphical interface.

---

## ✨ Features
- **Real-time Prediction**: Draw a digit on the canvas and get instant predictions.
- **CNN Model**: A trained deep learning model with **~99% test accuracy**.
- **User-Friendly GUI**: Clear canvas functionality and confidence score display.
- **Training Metrics**: Visualization of training/validation accuracy and loss.

---

## 🛠 Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/digit-recognizer.git
   cd digit-recognizer
Install Dependencies:

bash
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/handwritten-digit-recognizer.git
   cd handwritten-digit-recognizer
Install Dependencies:

bash
pip install -r requirements.txt
Train the Model (Required for First-Time Use):

bash
python train.py --epochs 15 --batch_size 128
This generates the model file (models/model.keras) and training logs.

Skip this step if using a pre-trained model.

Launch the GUI Application:

bash
python gui.py
Draw a digit on the canvas and click Recognize for predictions.

📂 Project Structure
.
├── models/             # Saved model checkpoints
├── logs/               # Training logs and metrics
├── gui.py              # GUI application code
├── train.py            # Model training script
├── requirements.txt    # Dependencies
└── README.md

📊 Logging & Model Saving
Automatic Logging: During training, logs are generated via TensorBoard and stored in logs/logs/. These include:

Training/validation accuracy and loss.

Histograms of layer weights.

Model Checkpoints: The best model (based on validation accuracy) is saved to models/my_model.keras.

Final Model: The trained model is saved as models/model.keras after training completes.

Training Metrics: A plot of accuracy and loss curves is saved to logs/training_metrics.png.

```markdown
# Handwritten Digit Recognition System 🔢

A complete implementation of a Convolutional Neural Network (CNN) for MNIST digit classification with a graphical user interface for real-time predictions. Built with TensorFlow and Tkinter. Ideal for ML education and prototyping.

![GUI Demo](https://via.placeholder.com/600x400?text=GUI+Preview+%7C+Draw+%26+Recognize+Digits) *Example: GUI Interface*

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/bb30fps/handwritten-digit-recognizer.git
cd handwritten-digit-recognizer
pip install -r requirements.txt
```

### 2. Train Model (First-Time Setup)
```bash
python train.py --epochs 15
```

### 3. Launch GUI
```bash
python gui.py
```

---

## ✨ Key Features

| Component         | Highlights                                                                 |
|-------------------|----------------------------------------------------------------------------|
| **CNN Model**     | 4-layer architecture with dropout, 99% test accuracy on MNIST              |
| **Smart GUI**     | Real-time predictions, confidence scores, canvas reset functionality       |
| **Training Suite**| Automatic model checkpointing, TensorBoard integration, metrics visualization |
| **Preprocessing** | Auto-inversion, normalization, and resizing for MNIST compatibility       |

---

## 📂 Project Structure

```plaintext
digit-recognizer/
├── models/               # Saved models
│   ├── model.keras      # Production model
│   └── my_model.keras   # Best validation model
├── logs/                # Training artifacts
│   ├── metrics/         # Accuracy/loss plots
│   └── tensorboard/     # Training logs for visualization
├── gui.py               # Graphical interface
├── train.py             # Model training script
└── requirements.txt     # Dependency specifications
```

---

## 🧠 Model Architecture

```python
Sequential(
    Conv2D(32, (5,5), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
)
```

---

## 📊 Training & Logging

### Automated Workflow
1. **Model Checkpoints**  
   - Saves best-performing model to `models/my_model.keras` during training
   - Criteria: Validation accuracy

2. **TensorBoard Integration**  
   ```bash
   tensorboard --logdir logs/tensorboard
   ```
   - Track metrics in real-time at `http://localhost:6006`

3. **Training Reports**  
   - Accuracy/loss plots auto-saved to `logs/metrics/training_metrics.png`

### Hyperparameters
| Parameter       | Value   | Description                          |
|-----------------|---------|--------------------------------------|
| Batch Size      | 128     | Samples per gradient update          |
| Base Learning Rate | 0.001 | Adam optimizer initial rate          |
| Early Stopping  | 3 epochs| Patience for validation loss         |

---

## 🖌️ GUI Usage Guide

1. **Drawing Tips**
   - Use mouse/touchpad to draw digits
   - Center drawings for best recognition
   - Black on white background preferred

2. **Workflow**
   ```mermaid
   graph LR
   A[Draw Digit] --> B[Preprocess Image]
   B --> C[Model Prediction]
   C --> D[Display Results]
   ```

3. **Preprocessing Steps**
   - Canvas capture → Grayscale → 28x28 resize → Color inversion → Normalization

---

## 🛠️ Troubleshooting

### Common Issues

| Symptom                          | Solution                                  |
|----------------------------------|-------------------------------------------|
| "Model not found" error          | Run `train.py` to generate model files    |
| Low prediction confidence        | Draw clearer, centered digits            |
| TensorFlow GPU errors            | Install CPU-only version with `tensorflow-cpu` |
| Dependency conflicts             | Use virtual environment                   |

### Advanced Configuration
```python
# To modify model architecture (train.py)
def create_model():
    model = Sequential([
        layers.Conv2D(64, (3,3), ...  # Increase filters
        layers.Dense(256, ...         # Larger dense layer
    ])
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

---

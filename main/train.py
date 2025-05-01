import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt


def create_model(input_shape=(28, 28, 1), num_classes=10):  # create and compile the cnn model
    model = models.Sequential([
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Train the model with callbacks and validation
def train_model(model, x_train, y_train, x_test, y_test, args):
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(args.model_dir, 'my_model.keras'),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(args.log_dir, 'logs'),
            histogram_freq=1
        )
    ]

    history = model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks_list,
        verbose=1
    )
    return history


def plot_training_history(history, save_path):  # Plot and save training metrics
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_metrics.png'))
    plt.close()


def main(args):
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Load and prepare data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize
    x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Create and train model
    model = create_model()
    model.summary()

    history = train_model(model, x_train, y_train, x_test, y_test, args)

    # Save final model
    model.save(os.path.join(args.model_dir, 'model.keras'))
    print(f"Model saved to {os.path.join(args.model_dir, 'model.keras')}")

    # Plot training history
    plot_training_history(history, args.log_dir)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST CNN Trainer')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for training logs')

    args = parser.parse_args()

    main(args)

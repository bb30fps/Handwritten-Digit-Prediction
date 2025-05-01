import tkinter as tk
from PIL import Image, ImageGrab, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


class DigitRecognizerApp:
    def __init__(self, root):
        
        self.root = root
        self.model = load_model('models/model.keras')
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Handwritten Digit Recognizer")

        self.canvas = tk.Canvas(self.root, width=300,
                                height=300, bg="white", cursor="cross")
        self.label = tk.Label(
            self.root, text="Draw a digit", font=("Helvetica", 24))
        self.btn_recognize = tk.Button(
            self.root, text="Recognize", command=self.predict_digit)
        self.btn_clear = tk.Button(
            self.root, text="Clear", command=self.clear_canvas)

        self.canvas.grid(row=0, column=0, pady=2, padx=2)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.btn_recognize.grid(row=1, column=1, pady=2)
        self.btn_clear.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw)

    def draw(self, event):
        x, y, r = event.x, event.y, 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.label.config(text="Draw a digit")

    def preprocess_image(self):
        # Get canvas coordinates
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        # Grab and process image
        img = ImageGrab.grab((x, y, x1, y1)).convert('L')
        img = img.resize((28, 28))
        img = ImageOps.invert(img)
        return np.array(img).reshape(1, 28, 28, 1) / 255.0

    def predict_digit(self):
        try:
            processed_img = self.preprocess_image()
            prediction = self.model.predict(processed_img)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)
            self.label.config(
                text=f"Digit: {digit}\nConfidence: {confidence*100:.2f}%")
        except Exception as e:
            self.label.config(text="Error in prediction")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

import tkinter as tk
from PIL import Image, ImageDraw
import io
import numpy as np

from backPropMNIST import NeuralNetwork

class DigitRecognizer:
    def __init__(self, model_path, input_size=(28, 28)):
        """Initialize the digit recognizer with a trained model"""
        self.input_size = input_size
        self.network = NeuralNetwork(path=f"{model_path}")
        

class DrawingCanvas:
    def __init__(self, root, recognizer):
        self.root = root
        self.recognizer = recognizer
        
        # Create canvas
        self.canvas = tk.Canvas(root, width=280, height=280, bg='black')
        self.canvas.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Create prediction label
        self.pred_label = tk.Label(root, text="Prediction: -", font=('Arial', 24))
        self.pred_label.pack(side=tk.TOP, padx=5, pady=5)
        
        # Create clear button
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.TOP, padx=5, pady=5)
        
        # Drawing state
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        self.counter = 0
        
    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                  fill='white', width=20, capstyle=tk.ROUND)
            self.last_x = x
            self.last_y = y
    
    def stop_drawing(self, event):
        self.drawing = False
        self.predict_digit()
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.pred_label.config(text="Prediction: -")
    
    def predict_digit(self):
        # Convert canvas to image
        x = self.canvas.winfo_width()
        y = self.canvas.winfo_height()
        image = Image.new('RGB', (x, y), 'black')
        draw = ImageDraw.Draw(image)

        # Draw all canvas items to the image
        items = self.canvas.find_all()
        for item in items:
            coords = self.canvas.coords(item)
            draw.line(coords, fill='white', width=15)

        # Convert to grayscale and resize to 28x28
        img = image.convert('L')
        img = img.resize((28, 28))
        
        
        # Convert to numpy array and normalize
        img_array = np.array(img).reshape(1, 784)
        img_array = img_array 

        # Get prediction
        predictions = self.recognizer.network.test(img_array, None)
        predicted_digit = np.argmax(predictions)
        confidence = predictions[0][predicted_digit]
        
        self.pred_label.config(text=f"Prediction: {predicted_digit} ({confidence:.2%})")
        
        # Save the image and prediction
        image.save(f'real_word_test/debug_digit_{self.counter}_{predicted_digit}.png')
        self.counter += 1

def launch_ui(model_path):
    root = tk.Tk()
    root.title("Digit Recognizer")
    
    recognizer = DigitRecognizer(model_path)
    canvas = DrawingCanvas(root, recognizer)
    
    root.mainloop()

# Add to your main block:
if __name__ == "__main__":
    # Launch UI after training
    model_path = f'models/two_layer_100_100_augmentation_remove_pixels/run_1/'
    launch_ui(model_path)

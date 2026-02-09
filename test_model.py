import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import pickle

# Load model and class names
model = tf.keras.models.load_model("skin_disease_model.keras")
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

img_size = (450, 450)

# Prediction logic
def predict_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]

    top_index = np.argmax(preds)
    top_class = class_names[top_index]
    top_prob = round(float(preds[top_index]) * 100, 2)

    return top_class, top_prob

# Upload button handler
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    # Load image
    pil_img = Image.open(file_path).resize((300, 300))
    tk_img = ImageTk.PhotoImage(pil_img)
    image_label.config(image=tk_img)
    image_label.image = tk_img

    # Predict and display result
    predicted_class, probability = predict_image(file_path)
    result_text.set(f"Predicted: {predicted_class} ({probability}%)")

# Create main window
root = tk.Tk()
root.title("Skin Disease Detection")
root.geometry("700x700")
root.configure(bg="#f7f9fc")

# Header
header = tk.Label(root, text="Skin Disease Detection", font=("Helvetica", 24, "bold"), bg="#f7f9fc", fg="#333")
header.pack(pady=30)

# Upload button
upload_btn = tk.Button(root, text="Upload an Image", command=upload_image,
                       font=("Helvetica", 14), bg="#4A90E2", fg="white", padx=20, pady=10, bd=0, relief="ridge", cursor="hand2")
upload_btn.pack(pady=10)

# Image display
image_label = tk.Label(root, bg="#f7f9fc")
image_label.pack(pady=20)

# Prediction result
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text,
                        font=("Helvetica", 16), bg="#ffffff", fg="#333333",
                        wraplength=500, justify="center", relief="solid", bd=1, padx=10, pady=10)
result_label.pack(pady=10)

# Start GUI
root.mainloop()

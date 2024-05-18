import tkinter as tk
from tkinter import filedialog, Label, Frame
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load mô hình đã huấn luyện
model = load_model('CNN_model.h5')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(256, 256))  # Resize image
    img_array = img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 256.0  # Normalize pixel values
    return img_array

def predict_pneumonia(img_path, model):
    preprocessed_image = preprocess_image(img_path)

    # Predict the class of the image
    prediction = model.predict(preprocessed_image)

    # Convert prediction to class label
    class_label = np.argmax(prediction, axis=1)

    if class_label[0] == 0:
        result = 'Không bị mắc bệnh viêm phổi'
    else:
        result = 'Bị mắc bệnh viêm phổi'

    return result


def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        result = predict_pneumonia(file_path, model)
        img = Image.open(file_path)
        img = img.resize((256, 256), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)

        panel.config(image=img)
        panel.image = img

        result_label.config(text="Kết quả: " + result)

# Tạo giao diện người dùng với Tkinter
root = tk.Tk()
root.title("Chẩn đoán viêm phổi từ ảnh X-quang")

frame = Frame(root, padx=10, pady=10)
frame.pack(padx=10, pady=10)

open_button = tk.Button(frame, text="Chọn ảnh", command=open_file)
open_button.grid(row=0, column=0, padx=10, pady=10)

result_label = tk.Label(frame, text="Kết quả: ", font=("Helvetica", 14))
result_label.grid(row=0, column=1, padx=10, pady=10)

panel = Label(frame)
panel.grid(row=1, column=0, columnspan=2, pady=10)

root.mainloop()
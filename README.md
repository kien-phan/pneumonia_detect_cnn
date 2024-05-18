# Chẩn đoán Viêm Phổi Sử Dụng CNN với Fine-Tuning VGG16

Repository này chứa mã nguồn để huấn luyện và đánh giá một mô hình Mạng Nơ-ron Tích Chập (CNN) nhằm phát hiện viêm phổi từ ảnh X-quang ngực sử dụng transfer learning với VGG16. Mô hình được huấn luyện và kiểm thử trên một tập dữ liệu ảnh X-quang ngực có sẵn trên Kaggle.

## Mục Lục
- [Cài Đặt](#cài-đặt)
- [Tập Dữ Liệu](#tập-dữ-liệu)
- [Huấn Luyện Mô Hình](#huấn-luyện-mô-hình)
- [Sử Dụng Mô Hình Để Dự Đoán](#sử-dụng-mô-hình-để-dự-đoán)
- [Cảm Ơn](#cảm-ơn)

## Cài Đặt

Truy cập [Kaggle](https://www.kaggle.com/), tạo 1 notebook và sử dụng Code trong tệp "train.py" để huấn luyện model. Sau khi có được model hoặc sử dụng model đã train sẵn tại [đây](https://drive.google.com/file/d/1uoAc6mANiFJzWtZRhtpqeafGILcTFjPT/view?usp=drive_link) và sử dụng code trong file "detect.py" để chẩn đoán xem 1 ảnh chụp x-quang của 1 người có bị mắc bệnh hay không.

## Tập Dữ Liệu

Tập dữ liệu sử dụng để huấn luyện và kiểm thử là tập dữ liệu Ảnh X-quang Ngực (Pneumonia) có sẵn trên Kaggle. Bạn có thể tải về từ [đây](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

Sau khi tải về, tập dữ liệu nên có cấu trúc như sau:

![image](https://github.com/kien-phan/pneumonia_detect_cnn/assets/89099392/1988226a-4ffd-4322-85ea-6f7a0e6bb84b)

## Huấn Luyện Mô Hình

Để huấn luyện mô hình, làm theo các bước sau:

1. **Tải tập dữ liệu lên Kaggle:**
   - Truy cập tài khoản Kaggle của bạn và tạo một tập dữ liệu mới với cấu trúc như trên.

2. **Chạy script huấn luyện trên Kaggle:**
   - Mở một Kaggle Notebook mới.
   - "Add input" dataset vào Kaggle Notebook.
   - Đảm bảo đặt đúng đường dẫn trong code tới tập dữ liệu trong notebook.

3. **Thực thi các cell trong notebook:**
   - Notebook bao gồm tất cả các bước cần thiết để tiền xử lý dữ liệu, huấn luyện mô hình, trực quan hóa kết quả huấn luyện và đánh giá mô hình.

## Sử Dụng Mô Hình Để Dự Đoán

Sau khi huấn luyện, bạn có thể sử dụng model để dự đoán viêm phổi trên các ảnh X-quang ngực mới.

1. **Tải mô hình đã huấn luyện:**
    ```python
    from tensorflow.keras.models import load_model
    model = load_model('pneumonia_detection_model.h5')
    ```

2. **Dự đoán trên một ảnh mới:**
    ```python
import tkinter as tk
from tkinter import filedialog, Label, Frame
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
    ```

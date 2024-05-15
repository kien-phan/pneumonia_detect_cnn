# Phát Hiện Viêm Phổi Sử Dụng CNN với Fine-Tuning VGG16

Repository này chứa mã nguồn để huấn luyện và đánh giá một mô hình Mạng Nơ-ron Tích Chập (CNN) nhằm phát hiện viêm phổi từ ảnh X-quang ngực sử dụng transfer learning với VGG16. Mô hình được huấn luyện và kiểm thử trên một tập dữ liệu ảnh X-quang ngực có sẵn trên Kaggle.

## Mục Lục
- [Cài Đặt](#cài-đặt)
- [Tập Dữ Liệu](#tập-dữ-liệu)
- [Huấn Luyện Mô Hình](#huấn-luyện-mô-hình)
- [Sử Dụng Mô Hình Để Dự Đoán](#sử-dụng-mô-hình-để-dự-đoán)
- [Cảm Ơn](#cảm-ơn)

## Cài Đặt

Truy cập [Kaggle](https://www.kaggle.com/), tạo 1 notebook và sử dụng Code trong tệp "main.py"

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

Sau khi huấn luyện, bạn có thể sử dụng mô hình để dự đoán viêm phổi trên các ảnh X-quang ngực mới.

1. **Tải mô hình đã huấn luyện:**
    ```python
    from tensorflow.keras.models import load_model
    model = load_model('pneumonia_detection_model.h5')
    ```

2. **Dự đoán trên một ảnh mới:**
    ```python
    import numpy as np
    from tensorflow.keras.preprocessing import image

    def predict_pneumonia(img_path, model):
        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        if prediction[0] > 0.5:
            return 'Phát hiện viêm phổi'
        else:
            return 'Không phát hiện viêm phổi'

    img_path = 'duong-dan-toi-anh-cua-ban.jpg'
    result = predict_pneumonia(img_path, model)
    print(result)
    ```

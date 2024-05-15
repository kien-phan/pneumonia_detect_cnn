# import thư viện
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2  # Thêm dòng này
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
sns.set(style="darkgrid", color_codes=True)

# Cấu hình các tham số
img_width, img_height = 256, 256
batchsize = 64
epochs = 50
num_of_class = 2

# Chuẩn bị dữ liệu với Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(
    '/kaggle/input/chest-xray-pneumonia/chest_xray/train',
    target_size=(img_width, img_height),
    batch_size=batchsize,
    class_mode='binary',  # Thay đổi class_mode thành 'binary'
    shuffle=False
)

validation = val_test_datagen.flow_from_directory(
    '/kaggle/input/chest-xray-pneumonia/chest_xray/val',
    target_size=(img_width, img_height),
    batch_size=batchsize,
    class_mode='binary'  # Thay đổi class_mode thành 'binary'
)

test = val_test_datagen.flow_from_directory(
    '/kaggle/input/chest-xray-pneumonia/chest_xray/test',
    target_size=(img_width, img_height),
    batch_size=batchsize,
    class_mode='binary'  # Thay đổi class_mode thành 'binary'
)

plt.pie([len(train), len(validation), len(test)],
        labels=['train', 'validation', 'test'], autopct='%.1f%%', colors=['orange', 'red', 'lightblue'], explode=(0.05, 0, 0))
plt.show()

# Cân bằng dữ liệu với SMOTE
x_train, y_train = [], []
for i in range(len(train)):
    x_train.extend(train[i][0])
    y_train.extend(train[i][1])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = x_train.reshape(x_train.shape[0], -1)  # Reshape for SMOTE
smote = SMOTE(random_state=42)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

x_train_res = x_train_res.reshape(-1, img_width, img_height, 3)  # Reshape back to image shape

# Định nghĩa mô hình CNN với fine-tuning thêm các lớp
def CNN_Model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    for layer in base_model.layers[:-6]:
        layer.trainable = False

    CNN = Sequential()
    CNN.add(Input(shape=(img_width, img_height, 3)))
    CNN.add(base_model)
    CNN.add(Conv2D(64, (3, 3), activation='relu'))
    CNN.add(MaxPooling2D(pool_size=(2, 2)))
    CNN.add(Flatten())
    CNN.add(Dropout(0.5))
    CNN.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    CNN.add(BatchNormalization())
    CNN.add(Dropout(0.5))
    CNN.add(Dense(1, activation='sigmoid'))  # Thay đổi thành 1 đơn vị đầu ra với sigmoid
    
    return CNN

# Huấn luyện mô hình
model = CNN_Model()
model.compile(optimizer=Adam(learning_rate=1e-6, clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)

hist = model.fit(
    x_train_res,
    y_train_res,
    validation_data=validation,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr]
)

# Trực quan hóa kết quả huấn luyện
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs_range = range(1, len(train_loss) + 1)

plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

train_accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']

plt.plot(epochs_range, train_accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Đánh giá mô hình trên tập dữ liệu kiểm tra
test.batch_size = 16
test_loss = []
test_accuracy = []
for i in range(len(test)):
    batch = test[i]
    loss, accuracy = model.evaluate(batch[0], batch[1], verbose=0)
    test_loss.append(loss)
    test_accuracy.append(accuracy)

# Tính toán loss và accuracy trung bình
test_loss = np.mean(test_loss)
test_accuracy = np.mean(test_accuracy)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Lưu mô hình dưới dạng tệp HDF5
model.save('cnn_model_final_v3.h5', overwrite=True)

# Dự đoán
from tensorflow.keras.preprocessing import image
def predict_pneumonia(img_path, model):
    # Load và xử lý ảnh
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normal hóa ảnh giống như khi huấn luyện

    # Dự đoán
    prediction = model.predict(img_array)
    
    if prediction[0] > 0.5:
        result = 'Bị mắc bệnh viêm phổi'
    else:
        result = 'Không bị mắc bệnh viêm phổi'
    
    return result

# Sử dụng hàm này để dự đoán cho một ảnh cụ thể
img_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person104_bacteria_492.jpeg'  # Thay đường dẫn bằng đường dẫn đến ảnh của bạn
result = predict_pneumonia(img_path, model)
print(result)
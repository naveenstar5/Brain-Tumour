# brain_tumor_all_in_one.py

import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# ====== CONFIGURATION ======
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
DATASET_DIR = 'dataset'       # Folder with your images: dataset/yes and dataset/no
MODEL_PATH = 'brain_model.h5' # Model will be saved here

# ====== TRAINING FUNCTION ======
def train_brain_model():
    if not os.path.exists(DATASET_DIR):
        print(f"{DATASET_DIR} not found. Please create dataset folder with subfolders for classes (yes/no).")
        return

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_data = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, validation_data=val_data, epochs=10)
    model.save(MODEL_PATH)
    print(f"Model saved as {MODEL_PATH}")
    return model

# ====== PREDICTION FUNCTION ======
def predict_brain_tumor(img_path):
    if not os.path.exists(MODEL_PATH):
        print(f"{MODEL_PATH} not found. Please train the model first.")
        return
    if not os.path.exists(img_path):
        print(f"{img_path} not found. Please provide a valid test image.")
        return

    model = load_model(MODEL_PATH)
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        print("Tumor Detected")
    else:
        print("No Tumor Detected")

# ====== MAIN FUNCTION ======
if __name__ == "__main__":
    print("===== Brain Tumor Detection =====\n")
    
    choice = input("Choose an option:\n1. Train Model\n2. Predict Image\nEnter 1 or 2: ")

    if choice == '1':
        train_brain_model()
    elif choice == '2':
        test_img = input("Enter path to test image: ")
        predict_brain_tumor(test_img)
    else:
        print("Invalid choice. Exiting.")

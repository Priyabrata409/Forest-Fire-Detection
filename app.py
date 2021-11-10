from sklearn import metrics
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
    array_to_img,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    classification_report,
)

X = []
y = []
d = {"fire_images": 1, "non_fire_images": 0}
for directory in os.listdir(
    "C:/Users/priyabrata/Forest_Fire_Detection/data/fire_dataset"
):
    for img in os.listdir(
        os.path.join(
            r"C:\Users\priyabrata\Forest_Fire_Detection\data\fire_dataset", directory
        )
    ):
        img_path = os.path.join(
            os.path.join(
                r"C:\Users\priyabrata\Forest_Fire_Detection\data\fire_dataset",
                directory,
            ),
            img,
        )
        image = cv2.imread(img_path)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (320, 320))
            X.append(image)
            y.append(d[directory])
        except:
            pass

X = np.array(X)
y = np.array(y)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, stratify=y)
train_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.9, 1.5],
    zoom_range=0.3,
)
test_datagen = ImageDataGenerator(rescale=1 / 255.0)

train_datagen.fit(train_X, augment=True)
train_data = train_datagen.flow(train_X, train_y)
test_datagen.fit(test_X)
test_data = test_datagen.flow(test_X, test_y)

## Model building
early_stop = EarlyStopping(monitor="val_accuracy", patience=5, verbose=2, mode="max")
model_check = ModelCheckpoint(
    "saved_model\model.h5",
    monitor="val_accuracy",
    verbose=2,
    save_best_only=True,
    mode="max",
)

model = Sequential(
    [
        Conv2D(32, 3, activation="relu", input_shape=(320, 320, 3)),
        MaxPooling2D(3, 3),
        Conv2D(64, 3, activation="relu"),
        MaxPooling2D(3, 3),
        Conv2D(128, 3, activation="relu"),
        MaxPooling2D(3, 3),
        Flatten(),
        Dense(100, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ]
)
print(model.summary())
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(
    train_data,
    epochs=100,
    validation_data=test_data,
    callbacks=[early_stop, model_check],
)

scores = model.history.history
plt.figure(figsize=(10, 6))
plt.plot(scores["accuracy"], label="train_accuracy")
plt.plot(scores["val_accuracy"], label="val_accuracy")
plt.legend()
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracies")
plt.savefig("Scores.png", bbox_inches="tight")
plt.show()

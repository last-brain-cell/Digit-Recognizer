import pandas as pd
import numpy as np
import random
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

X_train = np.array(train.drop(labels=['label'], axis=1))
y_train = np.array(train['label'])

X_test = np.array(test)

# Preprocessing the Train and Test Dataset as per CNN input requirements

# One Hot Encoding the Labels as the models last layer expects it
# I found out that this was the case through a gruelling debugging process XD
y_train = to_categorical(y_train, num_classes=10)

# Reshaping the datasets
X_train = X_train.reshape(len(X_train), 28, 28, 1)
y_train = y_train.reshape(len(y_train), 10)

X_test = X_test.reshape(len(X_test), 28, 28, 1)

# to ensure that the input data has a consistent range and distribution,
# which helps the network to learn better and faster.
X_train = X_train/255
X_test = X_test/255

# making sure that the all the shapes are correct
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)

# Visually Representing a random image

# idx = random.randint(0, len(X_train))
#
# img = X_train[idx]
#
# plt.figure(figsize=(2,2))
# plt.imshow(img, cmap='gray')

# model = Sequential([
#     Conv2D(32, (1,1), activation='relu',padding='same', input_shape=(28,28, 1)),
#     BatchNormalization(),

#     Conv2D(32, (3,3), activation='relu', padding='same'),
#     MaxPooling2D((2,2)),
#     Dropout(0.2),

#     Flatten(),
#     Dense(64, activation='relu'),
#     Dropout(0.3),
#     BatchNormalization(),
#     Dense(10, activation='softmax')
# ])

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((4, 4)),
    Dropout(0.1),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

datagen = ImageDataGenerator(
    rotation_range=10,  # rotate images randomly within 10 degrees
    zoom_range=0.1,     # zoom images randomly within 10%
    width_shift_range=0.1,  # shift images horizontally randomly within 10%
    height_shift_range=0.1, # shift images vertically randomly within 10%
    shear_range=0.1,    # shear images randomly within 10 degrees
    horizontal_flip=False, # flip images horizontally randomly
    vertical_flip=False   # flip images vertically randomly
)

# Fit the augmentation method on training set
datagen.fit(X_train)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

plot_model(model, show_shapes=True, show_layer_names=False,
           dpi=60, show_layer_activations=True, rankdir='TB')

# model.fit(X_train, y_train, epochs=10, batch_size=9) #7 epochs

# Train the model using data augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=12),
    epochs=15,
)

# plt.plot(model.history.history['accuracy'])
# plt.title('Accuracy Plot')

# plt.plot(model.history.history['loss'])
# plt.title('Loss Plot')

predictions = model.predict(X_test)

def result(prediction, idx):
    assert idx < len(X_test)
    assert idx >= 0
    pred = prediction[idx]
    print(f"Predicted Digit: {pred.argmax()}\n\n")
    print("Actual Digit: ")
    plt.imshow(X_test[idx], cmap='gray')
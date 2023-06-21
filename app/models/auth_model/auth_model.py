import rarfile
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import shutil
from keras.preprocessing.image import ImageDataGenerator
import random
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model
import os
import cv2
import random
import imgaug.augmenters as iaa


# Ruta de la carpeta que contiene las imágenes
dataset_path = '../../data/authentication/faces'

# Ruta de la carpeta "me"
me_folder = os.path.join(dataset_path, "me")

# Ruta de la carpeta "others"
others_folder = os.path.join(dataset_path, "others")

# Número objetivo de imágenes
target_count = 1000

# Obtener la lista de imágenes existentes en la carpeta "me"
me_images = os.listdir(me_folder)
existing_me_count = len(me_images)

# Obtener la lista de imágenes existentes en la carpeta "others"
others_images = os.listdir(others_folder)
existing_others_count = len(others_images)

# Calcular el número de imágenes adicionales que se deben generar
me_additional_count = target_count - existing_me_count
others_additional_count = target_count - existing_others_count

# Crear un objeto Sequential de imgaug con las transformaciones deseadas
seq = iaa.Sequential([
    iaa.Affine(rotate=(-20, 20)),  # Rotación aleatoria
    iaa.Fliplr(0.5),  # Volteo horizontal aleatorio
    iaa.GaussianBlur(sigma=(0, 1.0)),  # Desenfoque gaussiano aleatorio
    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))  # Ruido gaussiano aleatorio
])

# Generar imágenes adicionales para la carpeta "me"
for i in range(me_additional_count):
    # Seleccionar una imagen existente al azar
    image_name = me_images[i % existing_me_count]
    image_path = os.path.join(me_folder, image_name)

    # Cargar la imagen
    image = cv2.imread(image_path)

    # Aplicar transformaciones aleatorias a la imagen
    augmented_image = seq.augment_image(image)

    # Guardar la nueva imagen en la carpeta "me"
    new_image_name = f"me_augmented_{i}.jpg"
    new_image_path = os.path.join(me_folder, new_image_name)
    cv2.imwrite(new_image_path, augmented_image)

# Generar imágenes adicionales para la carpeta "others"
for i in range(others_additional_count):
    # Seleccionar una imagen existente al azar
    image_name = others_images[i % existing_others_count]
    image_path = os.path.join(others_folder, image_name)

    # Cargar la imagen
    image = cv2.imread(image_path)

    # Aplicar transformaciones aleatorias a la imagen
    augmented_image = seq.augment_image(image)

    # Guardar la nueva imagen en la carpeta "others"
    new_image_name = f"others_augmented_{i}.jpg"
    new_image_path = os.path.join(others_folder, new_image_name)
    cv2.imwrite(new_image_path, augmented_image)

# Reducir el número de imágenes en la carpeta "others" a 1000 eliminando imágenes de manera aleatoria
target_count_others = 1000
if len(others_images) > target_count_others:
    images_to_remove = random.sample(others_images, len(others_images) - target_count_others)
    for image in images_to_remove:
        image_path = os.path.join(others_folder, image)
        os.remove(image_path)

# Ruta del directorio raíz del dataset
dataset_dir = dataset_path

# Obtener la lista de imágenes de hombres
men_dir = os.path.join(dataset_dir, 'me')
men_images = os.listdir(men_dir)

# Obtener la lista de imágenes de mujeres
women_dir = os.path.join(dataset_dir, 'others')
women_images = os.listdir(women_dir)

# Contar el número de imágenes de hombres y mujeres
num_men_images = len(men_images)
num_women_images = len(women_images)

# Imprimir la información obtenida
print('Número de imágenes de mí:', num_men_images)
print('Número de imágenes de otros:', num_women_images)


# Directorio de los datos de entrenamiento
train_dir = dataset_path

# Dimensiones de las imágenes de entrada
input_shape = (128, 128, 3)

# Hiperparámetros del modelo
batch_size = 32
epochs = 10

# Preprocesamiento de los datos de entrenamiento
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Generar los generadores de imágenes de entrenamiento y validación
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Crear el modelo CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Guardar el modelo entrenado
model.save('authentication_model.h5')

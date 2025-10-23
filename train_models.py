"""
Pneumonia Detection Model Training
---------------------------------------------------
Trains two models:
1. Custom CNN
2. ResNet50 (Transfer Learning)
Includes evaluation and saving models.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- CONFIG ----------------
DATA_DIR = 'C:/Users/shyam/OneDrive/Desktop/P_Project/PneumoniaDetection_Project/chest_xray'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
MODEL_DIR = 'C:/Users/shyam/OneDrive/Desktop/P_Project/PneumoniaDetection_Project/models'
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Data Preparation ----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7,1.3],
    fill_mode='nearest'
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
val_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
test_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(train_gen.classes), y=train_gen.classes)
class_weights = dict(enumerate(class_weights))
print('Class Weights:', class_weights)

# ---------------- Model 1: Custom CNN ----------------
def build_custom_cnn(input_shape=(224,224,3)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', 
                      input_shape=input_shape, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.6),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

custom_model = build_custom_cnn()
custom_model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint(os.path.join(MODEL_DIR, 'custom_best.h5'), save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-6)
]

print('\nTraining Custom CNN...')
history_custom = custom_model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks
)

custom_model.save(os.path.join(MODEL_DIR, 'custom_CNN_model_final.h5'))
print('✅ Custom CNN model saved.')

# ---------------- Model 2: Transfer Learning (ResNet50) ----------------
def build_resnet(input_shape=(224,224,3), fine_tune_at=100):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(base.input, outputs)
    return model

resnet_model = build_resnet()
resnet_model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

callbacks_resnet = [
    ModelCheckpoint(os.path.join(MODEL_DIR, 'resnet_best.h5'), save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-6)
]

print('\nTraining ResNet50 Transfer Learning Model...')
history_resnet = resnet_model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks_resnet
)

resnet_model.save(os.path.join(MODEL_DIR, 'resnet_model_final.h5'))
print('✅ ResNet50 model saved.')

# ---------------- Evaluation ----------------
def evaluate_model(model, generator):
    preds = model.predict(generator)
    y_pred = (preds > 0.5).astype(int)
    y_true = generator.classes
    print(classification_report(y_true, y_pred, target_names=list(generator.class_indices.keys())))
    print('Confusion Matrix:\n', confusion_matrix(y_true, y_pred))

print('\nEvaluating Custom CNN:')
evaluate_model(custom_model, test_gen)

print('\nEvaluating ResNet50:')
evaluate_model(resnet_model, test_gen)

print('\n🎯 Training complete. Models saved in ./models/')

"""
Train Emotion Detection Model
Run this first to create accurate model!
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

print("🚀 Training Emotion Detection Model...")
print("👨‍💻 Made by: Shailendra Meghwal")

# Create model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create dummy data (replace with real dataset for production)
X_train = np.random.rand(100, 48, 48, 1).astype(np.float32)
y_train = keras.utils.to_categorical(np.random.randint(0, 7, 100), 7)

print("📊 Training (using sample data)...")
print("⚠️ For real accuracy, use FER2013 dataset")

# Train
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# Save
os.makedirs('model', exist_ok=True)
model.save('model/emotion_model.h5')
print("✅ Model saved to model/emotion_model.h5")
print("\n✨ Training complete! Now run: python utils/emotion_detector.py")
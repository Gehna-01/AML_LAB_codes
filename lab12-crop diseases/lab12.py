# Import libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Dataset path
train_dir = r"lab12-crop diseases\PlantVillage"
# Check if path exists
print("Folders:", os.listdir(train_dir))

# Image preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training data
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),   # slightly better than 32x32
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation data
val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# CNN Model
model = tf.keras.models.Sequential([
    
    tf.keras.layers.Input(shape=(64,64,3)),

    # Conv Block 1
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # Conv Block 2
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # Flatten
    tf.keras.layers.Flatten(),

    # Dense layers
    tf.keras.layers.Dense(64, activation='relu'),

    # Output
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=3   # increased slightly for better learning
)

# Evaluate
loss, accuracy = model.evaluate(val_data)

print("\nValidation Accuracy:", accuracy)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'path_to_your_data',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    'path_to_your_data',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',
    subset='validation'
)

# Build a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define a callback to monitor accuracy during training
class AccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1} - Accuracy: {logs['accuracy']:.4f} - Validation Accuracy: {logs['val_accuracy']:.4f}")

# Train the model with the accuracy callback
history = model.fit(train_generator, epochs=10, validation_data=validation_generator, callbacks=[AccuracyCallback()])

import cv2

# Load the trained model
model = tf.keras.models.load_model('path_to_your_model')

# Function to capture image from camera and make predictions
def predict_camera_image():
    cap = cv2.VideoCapture(0)  # 0 represents the default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (150, 150))
        normalized_frame = resized_frame / 255.0  # Normalize pixel values
        prediction = model.predict(tf.expand_dims(normalized_frame, axis=0))
        label = 'Dog' if prediction > 0.5 else 'Cat'
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start the camera
predict_camera_image()

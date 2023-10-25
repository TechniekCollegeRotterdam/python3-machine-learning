import tensorflow as tf
import numpy as np
import sys

print("Loading model...")
model = tf.keras.models.load_model("model.keras")

file_path = sys.argv[1]
print(f"Loading image from {file_path}...")
img = tf.keras.preprocessing.image.load_img(file_path, color_mode="grayscale", target_size=(28, 28))
img = tf.keras.preprocessing.image.img_to_array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)
print("Predicting...")
prediction = model.predict(img)
print(f"Prediction: {np.argmax(prediction)}")
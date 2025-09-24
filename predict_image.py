import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# === Load the trained model ===
model = load_model("cnn_model.h5")
print("✅ Model loaded.")

# === Define class label map (update based on your dataset) ===
label_map = {
    0: 'Class A',
    1: 'Class B',
    2: 'Class C',
    # Add all your actual class labels here
}

# === Load and preprocess the input image ===
img_path = "input_image.jpg"  # Replace with your test image file name
img = image.load_img(img_path, target_size=(32, 32))  # Make sure this matches your model input size

# Convert to array and normalize
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalize pixel values
img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 32, 32, 3)

# === Make prediction ===
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# === Show result ===
print("🔍 Predicted class index:", predicted_class)
print("🏷️ Predicted label:", label_map.get(predicted_class, "Unknown"))

# === Optional: Show the image ===
plt.imshow(img)
plt.title(f"Prediction: {label_map.get(predicted_class, 'Unknown')}")
plt.axis('off')
plt.show()

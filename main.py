import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# === LOAD DATA ===
def load_pickle_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

print("Loading dataset from .pickle files...")

# Load the dictionaries from pickle files
train_data = load_pickle_data("data/train/train.pickle")
val_data = load_pickle_data("data/train/valid.pickle")
test_data = load_pickle_data("data/train/test.pickle")

# Extract features and labels
X_train = train_data['features']
y_train = train_data['labels']

X_val = val_data['features']
y_val = val_data['labels']

X_test = test_data['features']
y_test = test_data['labels']

# === PREPROCESSING ===
print("Preprocessing data...")

# Normalize image pixel values
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train)
y_val_cat = to_categorical(y_val)
y_test_cat = to_categorical(y_test)

# === BUILD MODEL ===
print("Building CNN model...")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_train_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === TRAIN MODEL ===
print("Training the model...")

history = model.fit(
    X_train, y_train_cat,
    batch_size=32,
    epochs=10,
    validation_data=(X_val, y_val_cat)
)

# === EVALUATE MODEL ===
print("Evaluating on validation data...")

val_loss, val_acc = model.evaluate(X_val, y_val_cat)
print(f"Validation Accuracy: {val_acc:.4f}")

# === PLOT ACCURACY ===
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# === CONFUSION MATRIX ===
print("Generating confusion matrix...")

y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
y_val_true = np.argmax(y_val_cat, axis=1)

cm = confusion_matrix(y_val_true, y_val_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

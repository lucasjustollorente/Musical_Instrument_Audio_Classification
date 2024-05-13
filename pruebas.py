import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import wavfile

# Step 1: Prepare the Dataset
src_root = 'clean'
audio_files = [os.path.join(src_root, file) for file in os.listdir(src_root) if file.endswith('.wav')]

# Assuming each folder in 'clean' corresponds to a different class
labels = [folder for folder in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, folder))]

# Split the data into training and validation sets
train_files, val_files = train_test_split(audio_files, test_size=0.1, random_state=42)

# Step 2: Extract Features using VGGish
vggish_model_url = "https://tfhub.dev/google/vggish/1"
vggish_model = hub.load(vggish_model_url)

def extract_features(audio_files):
    features = []
    for file in audio_files:
        sample_rate, audio_data = wavfile.read(file)
        # Normalize audio data to range [-1, 1]
        audio_data = audio_data / np.max(np.abs(audio_data))
        # Extract features using VGGish
        vggish_features = vggish_model(audio_data)
        features.append(vggish_features.numpy())
    return np.array(features)

train_features = extract_features(train_files)
val_features = extract_features(val_files)

# Step 3: Build and Train the Classification Model
num_classes = len(labels)
num_features = train_features.shape[1]  # Number of features extracted by VGGish

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(num_features,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Assuming you have labels encoded as integers
train_labels = [labels.index(os.path.split(os.path.dirname(file))[-1]) for file in train_files]
val_labels = [labels.index(os.path.split(os.path.dirname(file))[-1]) for file in val_files]

model.fit(train_features, train_labels, epochs=10, batch_size=32, validation_data=(val_features, val_labels))

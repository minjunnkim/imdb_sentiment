from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout

import numpy as np

import matplotlib.pyplot as plt

import sys
import os

# Get the absolute path of the parent directory
parent_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Add the parent directory to the system path
if parent_directory not in sys.path:
    sys.path.append(parent_directory)


# Load processed data
processed_data_directory = os.path.join(parent_directory, "data\\processed")

train_padded = np.load(os.path.join(processed_data_directory, "train_padded.npy"))
train_labels = np.load(os.path.join(processed_data_directory, "train_labels.npy"))
test_padded = np.load(os.path.join(processed_data_directory, "test_padded.npy"))
test_labels = np.load(os.path.join(processed_data_directory, "test_labels.npy"))


# Model Selection
model = Sequential([
    Embedding(input_dim=10000, output_dim=64),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Build the model by providing the input shape
model.build(input_shape=(None, 120))

# Display the model summary
model.summary()


# Train the model
history = model.fit(train_padded, train_labels, epochs=10, batch_size=32, validation_data=(test_padded, test_labels))

# Save the model
model_directory = os.path.join(parent_directory, "app\\models\\imdb_sentiment.h5")

model.save(model_directory)


# Plot training history
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
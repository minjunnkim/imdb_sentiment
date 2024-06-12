import sys
import os

# Get the absolute path of the parent directory
parent_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Add the parent directory to the system path
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

import pickle

from app.utils.preprocessing import clean_text, preprocess_data


# Load IMDb dataset
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']


# Preprocess data
train_sentences, train_labels = preprocess_data(train_data)
test_sentences, test_labels = preprocess_data(test_data)

# Balance the dataset
from collections import Counter

counter = Counter(train_labels)
print(f"Original dataset: {counter}")

# Create a balanced dataset by undersampling the majority class
min_count = min(counter.values())
balanced_train_sentences = []
balanced_train_labels = []

pos_count = 0
neg_count = 0

for sentence, label in zip(train_sentences, train_labels):
    if label == 1 and pos_count < min_count:
        balanced_train_sentences.append(sentence)
        balanced_train_labels.append(label)
        pos_count += 1
    elif label == 0 and neg_count < min_count:
        balanced_train_sentences.append(sentence)
        balanced_train_labels.append(label)
        neg_count += 1

print(f"Balanced dataset: {Counter(balanced_train_labels)}")

train_sentences = balanced_train_sentences
train_labels = balanced_train_labels


# Clean text data
train_sentences = [clean_text(sent) for sent in train_sentences]
test_sentences = [clean_text(sent) for sent in test_sentences]


# Tokenization and Padding
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding='post', maxlen=120)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, padding='post', maxlen=120)


# Save Tokenizer
tokenizer_save_path = os.path.join(parent_directory, "app\\tokenizer\\tokenizer.pickle")

with open(tokenizer_save_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Save processed data
processed_data_directory = os.path.join(parent_directory, "data\\processed")

np.save(os.path.join(processed_data_directory, "train_padded.npy"), train_padded)
np.save(os.path.join(processed_data_directory, "train_labels.npy"), train_labels)
np.save(os.path.join(processed_data_directory, "test_padded.npy"), test_padded)
np.save(os.path.join(processed_data_directory, "test_labels.npy"), test_labels)

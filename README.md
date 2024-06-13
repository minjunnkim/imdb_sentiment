# Sentiment Analysis Project (imdb_sentiment)

## Overview

This project aims to develop a sentiment analysis model that classifies movie reviews into positive or negative sentiments. The model is built using a Bidirectional Long Short-Term Memory (BiLSTM) neural network with pre-trained GloVe embeddings. The project includes data preprocessing, model training, evaluation, and deployment using FastAPI.

## Project Structure
```plaintext
imdb_sentiment/
│
├── app/
│ ├── init.py
│ ├── main.py
│ ├── models/
│ │ ├── model.h5
│ ├── tokenizer/
│ │ ├── tokenizer.pickle
│ ├── routes/
│ │ ├── init.py
│ │ ├── sentiment.py
│ ├── utils/
│ │ ├── init.py
│ │ ├── preprocessing.py
│
├── data/
│ ├── raw/
│ ├── processed/
│ │ ├── train_padded.npy
│ │ ├── train_labels.npy
│ │ ├── test_padded.npy
│ │ ├── test_labels.npy
│
├── glove.6B/
│
├── notebooks/
│ ├── data_preprocessing.ipynb
│ ├── model_training.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Objectives

- Develop a sentiment analysis model to classify movie reviews.
- Ensure the model is trained on a balanced dataset to mitigate bias.
- Deploy the model using FastAPI to provide a REST API for sentiment prediction.

## Data Preprocessing

1. **Data Loading**:
   - The IMDb movie reviews dataset is used, loaded via TensorFlow Datasets.

2. **Cleaning**:
   - Text data is cleaned by removing special characters and converting to lowercase.

3. **Tokenization**:
   - Text is tokenized into sequences of integers using the Keras `Tokenizer`.

4. **Padding**:
   - Sequences are padded to ensure uniform length, suitable for batch processing.

5. **Balancing**:
   - The dataset is balanced by undersampling the majority class to ensure equal representation of positive and negative samples.

6. **Saving**:
   - The tokenizer and processed data (padded sequences and labels) are saved for future use.

## Model Architecture

The model is a Bidirectional LSTM (BiLSTM) neural network with the following layers:

1. **Embedding Layer**:
   - Uses pre-trained GloVe embeddings to convert words into dense vectors of fixed size, capturing semantic meanings.

2. **Bidirectional LSTM Layers**:
   - First Bidirectional LSTM layer with 128 units and `return_sequences=True` to output sequences.
   - Dropout layer with 0.5 rate to prevent overfitting.
   - Second Bidirectional LSTM layer with 64 units for further capturing context from both directions.

3. **Dense Layers**:
   - A dense layer with 64 units and ReLU activation.
   - Output dense layer with 1 unit and sigmoid activation to predict the sentiment probability.

## Model Compilation and Training

- **Loss Function**: Binary Crossentropy, suitable for binary classification tasks.
- **Optimizer**: Adam, which adjusts learning rate dynamically.
- **Metrics**: Accuracy, to measure the performance of the model.
- **Training**: The model is trained on the balanced dataset for 10 epochs with a batch size of 32.

## Model Evaluation

The model’s performance is evaluated on the test set using accuracy and loss metrics. Additional evaluation metrics such as precision, recall, and F1-score are used to gain insights into model performance. Confidence levels for predictions are calculated to understand the model's certainty in its predictions.

## Deployment

The trained model is deployed using FastAPI, allowing it to serve predictions via a REST API. An endpoint `/predict` is created to accept text input and return the predicted sentiment and confidence level.

### FastAPI Endpoint

**Endpoint**: `/predict`
- **Input**: JSON with a text field, e.g., `{"text": "I love this movie!"}`
- **Output**: JSON with predicted sentiment and confidence level, e.g., `{"sentiment": "positive", "confidence": 0.9677}`

## Usage Instructions

### Installing GloVe Embeddings

1. **Download Zip file from Github:**
    - Link to GloVe Github Repository: https://github.com/stanfordnlp/GloVe
    - Link to GloVe Embeddings download: https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip 

2. **Moving the .txt files to glove.6B folder**
    - Please move the .txt files in the Zip file downloaded inside the project's glove.6B folder.

### Setting Up the Environment

1. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt

### Running the Jupyter Notebooks

1. **Navigate to the project directory:**
    ```bash
    cd imdb_sentiment

2. **Run the preprocessing notbook:**
    - Open and run `data_preprocessing.ipynb` to preprocess the data and save the processed data files.

3. **Run the training notebook:**
    - Open and run `model_training.ipynb` to train the model and save the trained model file.

### Changing Hyperparameters in Jupyter Notebooks

To tailor the model to your specific needs, you can change various hyperparameters in the `model_training.ipynb` notebook. Some of the key hyperparameters you might want to adjust include:
- Number of LSTM Units: Adjust the number of units in the LSTM layers.
- Dropout Rate: Change the dropout rate to prevent overfitting.
- Embedding Dimension: Modify the size of the word embeddings.
- Batch Size: Experiment with different batch sizes during training.
- Number of Epochs: Increase or decrease the number of epochs to control the training duration.

### Running the FastAPI Application

1. **Navigate to the project directory:**
    ```bash
    cd imdb_sentiment

2. **Run the FastAPI application using Uvicorn:**
    ```bash
    uvicorn app.main:app --reload

3. **Access the API documentation:**
    - Swagger UI: http://127.0.0.1:8000/docs
    - ReDoc: http://127.0.0.1:8000/redoc

## Additional Information

- Advanced Models: 
    - The model_training.ipynb notebook includes steps for training more advanced models like BERT, but these models are not used in the current API implementation.
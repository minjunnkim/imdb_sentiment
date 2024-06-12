from fastapi import APIRouter, Request
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

router = APIRouter()

# Load the trained model
model = tf.keras.models.load_model('app/models/imdb_sentiment.keras')

# Load the tokenizer
with open('app/tokenizer/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

class TextRequest(BaseModel):
    text: str

@router.post("/predict")
async def predict(request: TextRequest):
    text = request.text
    
    # Preprocess the text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=120, padding='post')
    
    # Predict sentiment
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = 'positive' if prediction > 0.5 else 'negative'
    confidence = float(prediction) if sentiment == 'positive' else float(1 - prediction)
    
    return {"sentiment": sentiment, "confidence": confidence}
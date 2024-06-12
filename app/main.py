from fastapi import FastAPI
from app.routes import sentiment

app = FastAPI()

app.include_router(sentiment.router)

@app.get("/")
def read_root():
    return {"message": "Hello, welcome to my IMDB Sentiment project. My model is trained on IMDB reviews from TensorFlow's datasets (TFDS). The model is a Bidirectional LSTM (Long Short-Term Memory) neural network with Embedding Layer made of pre-trained GloVe embeddings, two Bidirectional LSTM layers, first with 128 units and second with 64 units and with a dropout layer of 0.5 rate in between them to prevent overfitting, and lastly a Dense layer with 64 units and ReLU activation, and an output dense layer with 1 unit and sigmoid activation to predict the sentiment probability. Go to http://127.0.0.1:8000/docs to try it out."}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
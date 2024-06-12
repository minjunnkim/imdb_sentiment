import re

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    text = text.lower()
    return text

def preprocess_data(data):
    sentences = []
    labels = []

    for s, l in data:
        sentences.append(s.numpy().decode('utf8'))
        labels.append(l.numpy())

    return sentences, labels
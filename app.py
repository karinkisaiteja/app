import random
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import load_model
import streamlit as st
import os

# Load the pre-trained model and other data
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents1.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Get the current working directory
cwd = os.path.dirname(os.path.abspath(__file__))

# Path to the nltk_data directory
nltk_data_path = os.path.join(cwd, 'nltk_data')

def unzip_nltk_data(nltk_data_path):
    wordnet_zip_path = os.path.join(nltk_data_path, 'corpora', 'wordnet.zip')
    punkt_zip_path = os.path.join(nltk_data_path, 'tokenizers', 'punkt.zip')

    if not os.path.exists(os.path.join(nltk_data_path, 'wordnet')):
        with zipfile.ZipFile(wordnet_zip_path, 'r') as zip_ref:
            zip_ref.extractall(nltk_data_path)

    if not os.path.exists(os.path.join(nltk_data_path, 'punkt')):
        with zipfile.ZipFile(punkt_zip_path, 'r') as zip_ref:
            zip_ref.extractall(nltk_data_path)

# Unzip the WordNet and Punkt data files
unzip_nltk_data(nltk_data_path)

# Add the path to the NLTK data
nltk.data.path.append(nltk_data_path)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def main():
    st.title("Chatbot")
    user_message = st.text_input("You: ", "")
    if user_message:
        ints = predict_class(user_message, model)
        res = get_response(ints, intents)
        st.write("Bot: " + res)

if __name__ == '__main__':
    main()

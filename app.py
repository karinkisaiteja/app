import random
import spacy
from spacy.lang.en import English
import json
import pickle
import numpy as np
from keras.models import load_model
import streamlit as st

# Load the pre-trained model and other data
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents1.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Initialize the spaCy English language model
nlp = spacy.load("en_core_web_sm")

def clean_up_sentence(sentence):
    doc = nlp(sentence)
    sentence_words = [token.lemma_.lower() for token in doc]
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
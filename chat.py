import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.keras.models import load_model
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence = nltk.word_tokenize(sentence)
    lemmatized_words = [lemmatizer.lemmatize(i) for i in sentence]
    return lemmatized_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(message):
    intents_list = predict_class(message)
    intents_json = intents
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tags'] == tag:
            result = random.choice(i['response'])
            break
    return result

name = "Sam"


if __name__ == "__main__":
    while True:
        message = input("")
        if message == 'quit':
            break

        res = get_response(message)
        print(res)
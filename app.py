
#!/usr/bin/env python
# coding: utf-8

import random
import nltk
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request
import requests


with open('./27.it_context.json', 'r', encoding='utf-8') as f:
  BOT_CONFIG = json.load(f)

def clean(text):
  text = text.lower()
  cleaned_text = ''
  for ch in text:
    if ch in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя':
      cleaned_text = cleaned_text + ch
  return cleaned_text

def get_intent(text):
  for intent in BOT_CONFIG['intents'].keys():
    for example in BOT_CONFIG['intents'][intent]['examples']:
      cleaned_example = clean(example)
      cleaned_text = clean(text)
      if nltk.edit_distance(cleaned_example, cleaned_text) / max(len(cleaned_example), len(cleaned_text)) * 100 < 40:
        return intent
  return 'unknown_intent'

# ## Обучение модели

X = []
y = []
for intent in BOT_CONFIG['intents']:
    for example in BOT_CONFIG['intents'][intent]['examples']:
        X.append(example)
        y.append(intent)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3)) #CountVectorizer(analyzer='char', ngram_range=(1,3), preprocessor=clean)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


clf = RidgeClassifier() #LogisticRegression()
clf.fit(X_train_vectorized, y_train)
clf.score(X_train_vectorized, y_train), clf.score(X_test_vectorized, y_test)


def get_intent_by_model(text):
  vectorized_text = vectorizer.transform([text])
  return clf.predict(vectorized_text)[0]


def bot(text):
  return random.choice(BOT_CONFIG['intents'][get_intent_by_model(text)]['responses'])





app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def post_bot_response():
    userText = request.args.get('msg')
    return (bot(userText))



if __name__ == "__main__":
    app.run(host = '0.0.0.0', port=5400)#(host = '0.0.0.0', port=5100)

import numpy as np
from numpy.lib.function_base import vectorize
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import re
import pickle

labels = ['negative', 'positive']

# loading
with open('data/vectorizer.pk', 'rb') as handle:
    vectorizer = pickle.load(handle)


input_dim = 5000
model1=Sequential()
model1.add(Dense(25, input_dim=input_dim,activation='relu'))
model1.add(Dense(2, activation='softmax'))
model1.compile(loss='categorical_crossentropy',optimizer='adam')

model1.load_weights("data/model1.hdf5")

test_pred = ['I really love sentiment analysis. It is the best and I am so happy']


test_pred_array = vectorizer.transform(test_pred)
test_pred_array = test_pred_array.toarray()



prediction = model1.predict(test_pred_array)
prediction = np.argmax(prediction)
labels[prediction]


def get_predictions(text, model):
    text = vectorizer.transform([text]).toarray()
    prediction = model.predict(text)[0]
    return {
        'negative': prediction[0],
        'positive': prediction[1]
    }

test = get_predictions('I really love sentiment analysis. It is the best and I am so happy', model1)

test
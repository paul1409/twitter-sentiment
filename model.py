from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import numpy as np
import pandas as pd
import tensorflow as tf

data = pd.read_csv('static/models/twitter_training.csv', names=[
                   "Tweet_ID", "Entity", "Sentiment", "Text"])
data = data[['Text', 'Sentiment']]
data = data[data.Sentiment != "Neutral"]
data = data[data.Sentiment != "Irrelevant"]
data.Text = data.Text.apply(lambda x: str(x).lower())
data.Text = data.Text.apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatiser = WordNetLemmatizer()
stopwords = set(stopwords.words())


def remove_stopwords(ls):
    ls = [lemmatiser.lemmatize(word) for word in ls if word not in (
        stopwords) and (word.isalpha())]
    ls = " ".join(ls)
    return ls


data.Text = data.Text.apply(word_tokenize)
data.Text = data.Text.apply(remove_stopwords)

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_features = 1000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data.Text.values)
X = tokenizer.texts_to_sequences(data.Text.values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196

model = Sequential([
    Embedding(max_features, embed_dim, input_length=X.shape[1]),
    SpatialDropout1D(0.4),
    LSTM(lstm_out, dropout=0.2),
    Dense(2, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Y = pd.get_dummies(data.Sentiment).values
# X_train, X_test, Y_train, Y_test = train_test_split(
#     X, Y, test_size=0.33, random_state=42)


def setup():
    model.load_weights('static/models/weights.hdf5')
    # model.fit(X_train, Y_train, epochs=10, batch_size=32,
    #           callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=4)], verbose=1)


def apply_prediction(twt):
    twtData = tokenizer.texts_to_sequences([twt])
    twtData = pad_sequences(twtData, maxlen=99, dtype='int32', value=0)
    # print(twtData)
    sentiment = model.predict(twtData, batch_size=1, verbose=2)[0]
    sentimentValue = "negative" if(np.argmax(sentiment) == 0) else "positive"
    return sentimentValue



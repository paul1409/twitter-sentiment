from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pickle

labels = ['negative', 'positive']

# loading
with open('static/models/vectorizer.pk', 'rb') as handle:
    vectorizer = pickle.load(handle)


input_dim = 5000
model1=Sequential()
model1.add(Dense(25, input_dim=input_dim,activation='relu'))
model1.add(Dense(2, activation='softmax'))
model1.compile(loss='categorical_crossentropy',optimizer='adam')
model1.load_weights("static/models/model1.hdf5")

model2=Sequential()
model2.add(Dense(25, input_dim=input_dim,activation='relu'))
model2.add(Dense(50, input_dim=input_dim,activation='relu'))
model2.add(Dense(25, input_dim=input_dim,activation='relu'))
model2.add(Dense(10, input_dim=input_dim,activation='relu'))
model2.add(Dense(2, activation='softmax'))
model2.compile(loss='categorical_crossentropy',optimizer='adam')
model2.load_weights("static/models/model2.hdf5")

with open("static/models/model4.pkl", 'rb') as file:
    model4 = pickle.load(file)




# test_pred = ['I really love sentiment analysis. It is the best and I am so happy']


# test_pred_array = vectorizer.transform(test_pred).toarray()
# test_pred_array = test_pred_array.toarray()



# prediction = model1.predict(test_pred_array)
# prediction = np.argmax(prediction)
# labels[prediction]


def get_predictions(text, model):
    text = vectorizer.transform([text]).toarray()
    prediction = model.predict(text)[0]
    return {
        'negative': prediction[0],
        'positive': prediction[1]
    }

text = 'I really love sentiment analysis. It is the best and I am so happy'
pred_model1 = get_predictions(text, model1)
pred_model2 = get_predictions(text, model2)

# logistic regression
text = vectorizer.transform([text]).toarray()
pred_model4 = model4.predict(text)




import nltk
import tensorflow as tf
import numpy as np
import random
import json
from nltk.stem import PorterStemmer

nltk.download('punkt')
stemmer = PorterStemmer()

with open('intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore = ['?','!',',']
# loop through each sentence in the intent's patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each and every word in the sentence
        w = nltk.word_tokenize(pattern)
        # add word to the words list
        words.extend(w)
        # add word(s) to documents
        documents.append((w, intent['tag']))
        # add tags to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Perform stemming and lower each word as well as remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))

# remove duplicate classes
classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)

# create training data
training = []
output = []
# create an empty array for output
output_empty = [0] * len(classes)

# create training set, bag of words for each sentence
for doc in documents:
    # initialize bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stemming each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is '1' for current tag and '0' for rest of other tags
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffling features and turning it into np.array
random.shuffle(training)
training = np.array(training, dtype='object')

# creating training lists
train_x = list(training[:,0])
train_y = list(training[:,1])

from keras import regularizers

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(20, input_shape=(len(train_x[0]),),activation='relu'))
model.add(tf.keras.layers.Dense(10, kernel_regularizer=regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))
model.compile(tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=8, verbose=1)
model.save("model.pkl")

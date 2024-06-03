import nltk
import pickle
import os
import numpy as np
import json
import random
from keras.models import load_model
from nltk.stem import PorterStemmer

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

data = pickle.load( open( "training_data", "rb" ) )
model = load_model("model.pkl")
words = data['words']
classes = data['classes']
pickle.dump( {'words':words, 'classes':classes}, open( "training_data", "wb" ) )

def clean_up_sentence(sentence):
    # tokenizing the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stemming each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# returning bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words):
    # tokenizing the pattern
    sentence_words = clean_up_sentence(sentence)
    # generating bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    bag=np.array(bag)
    return(bag)

ERROR_THRESHOLD = 0.30
def classify(sentence):
    # generate probabilities from the model
    bag = bow(sentence, words)
    results = model.predict(np.array([bag]))
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results[0]) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def save_comment(username, comment):
    folder_name = 'Comments'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    filename = os.path.join(folder_name, f"{username}_comments.txt")
    with open(filename, 'a') as file:
        file.write(comment + '\n')


user_name = input("Masukkan nama Anda: ")
exited = False
default_found = False

while not default_found:
    user_input = "CuBot"
    results = classify(user_input)
    if results:
        for i in intents['intents']:
            if results[0][0] == i['tag'] == 'Default':
                print(f"{'CuBot':<{len(user_name)}}: {random.choice(i['responses'])}")
                default_found = True
                break

while not exited:
    user_input = input(f"{user_name}: ")
    results = classify(user_input)

    if results:
        for i in intents['intents']:
            if results[0][0] == i['tag'] == 'Berpisah':
                print(f"{'CuBot':<{len(user_name)}}: {random.choice(i['responses'])}")
                exited = True
                break
            elif results[0][0] == i['tag'] == 'CuBot_Minta_Komentar':
                comment = input(f"{'CuBot':<{len(user_name)}}: {random.choice(i['responses'])}\nKomentar Anda terkait CuBot: ")
                save_comment(user_name, comment)
                user_input = "Komentarnya sudah saya berikan yah"
                results = classify(user_input)
                for j in intents['intents']:
                    if j['tag'] == 'CuBot_Berterimakasih_Komentar':
                        print(f"{'CuBot':<{len(user_name)}}: {random.choice(j['responses'])}")
                        break
                break
            elif results[0][0] == i['tag'] == 'Pola_Tidak_Ditemukan':
                print(f"{'CuBot':<{len(user_name)}}: {random.choice(i['responses'])}")
                break
            elif results[0][0] == i['tag']:
                print(f"{'CuBot':<{len(user_name)}}: {random.choice(i['responses'])}")
                break
    else:
        for i in intents['intents']:
            if i['tag'] == 'Pola_Tidak_Ditemukan':
                print(f"CuBot: {random.choice(i['responses']):<{len(user_name)}}")
                break


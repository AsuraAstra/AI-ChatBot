import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers.legacy import SGD

lemmatizer = WordNetLemmatizer()

# Load the intents.json file
intents = json.loads(open("E:/Programs/aichatbot/intents.json").read())

# Create empty lists
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Loop through each sentence in the intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        # Add the word to the words list
        words.extend(word_list)
        # Add the word/sentence to the documents list
        documents.append((word_list, intent['tag']))
        # Add the tag to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lowercase each word and remove duplicates
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))


# Sort classes
classes = sorted(list(set(classes)))

# Print the length of the documents, classes and words
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('E:/Programs/aichatbot/words.pkl', 'wb')) # Save the words list
pickle.dump(classes, open('E:/Programs/aichatbot/classes.pkl', 'wb')) # Save the classes list

# Create the training data
training = []
# Create an empty array for the output
output_empty = [0] * len(classes)

# Create the training set, bag of words for each sentence
for document in documents:
    bag = []
    # List of tokenized words for the pattern
    word_patterns = document[0]
    # Lemmatize each word - create base word, in attempt to represent related words
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # Create the bag of words array with 1, if word match found in current pattern
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # Output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Shuffle the features and make numpy array
random.shuffle(training)
training = np.array(training)

# Create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) # First hidden layer
model.add(Dropout(0.5)) # Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting
model.add(Dense(64, activation='relu')) # Second hidden layer
model.add(Dropout(0.5)) # Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting
model.add(Dense(len(train_y[0]), activation='softmax')) # Output layer

# Compile the model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
SGD = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=['accuracy'])

# Fit the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('E:/Programs/aichatbot/chatbot_model.h5', hist)

print("Done")



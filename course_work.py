import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

dataset = pd.read_csv('jsbach_chorals_harmony.data',
                      delimiter=',',
                      header=None,
                      names=['Choral ID', 'Event number',
                             'Pitch Class C', 'Pitch Class C#/Db',
                             'Pitch Class D', 'Pitch Class D#/Eb',
                             'Pitch Class E', 'Pitch Class F',
                             'Pitch Class F#/Gb', 'Pitch Class G',
                             'Pitch Class G#/Ab', 'Pitch Class A',
                             'Pitch Class A#/Bb', 'Pitch Class B',
                             'Bass', 'Meter', 'Chord Label'])



# print(dataset.sample(5, random_state=0))
#
# sns.pairplot(dataset, hue='Chord Label', height=5)
# plt.show()
#
# sns.pairplot(dataset, hue='Bass', height=5)
# plt.show()
#
# sns.pairplot(dataset, hue='Meter', height=5)
# plt.show()
#
# sns.pairplot(dataset, hue='Event number', height=5)
# plt.show()

# split the dataset into input features (X) and target variable (y)
X_pitch = dataset.iloc[:, 2:15].replace({'YES': 1, 'NO': 0})
X_bass = pd.get_dummies(dataset.iloc[:, 15], prefix='Bass')
X_meter = dataset.iloc[:, 16]
X = pd.concat([X_pitch, X_bass, X_meter], axis=1)
X = pd.get_dummies(X)

y = dataset.iloc[:, 16]   # column 17 is the chord name

# perform one-hot encoding on the target variable (chord name)
y = pd.get_dummies(y)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

archs = ['Feed-Forward', 'MLP 1 layer', 'MLP 4 layers']

losses = []
accuracies = []


# define the feedforward neural network model
model = Sequential()
model.add(Dense(64, input_dim=147, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(102, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model to the training data
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)

# evaluate the model on the testing data
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
losses.append(loss)
accuracies.append(accuracy)
# print(f'Test loss: {loss:.3f}')
# print(f'Test accuracy: {accuracy:.3f}')


model = Sequential()
model.add(Dense(64, activation='relu', input_dim=147)) # input layer with 147 features
model.add(Dense(32, activation='relu')) # hidden layer with 32 units
model.add(Dense(102, activation='sigmoid')) # output layer with 102 units for the classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# evaluate the model on the testing data
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
losses.append(loss)
accuracies.append(accuracy)


model = Sequential()
model.add(Dense(64, activation='relu', input_dim=147)) # input layer with 147 features
model.add(Dense(32, activation='relu')) # hidden layer with 32 units
model.add(Dense(16, activation='relu')) # hidden layer with 32 units
model.add(Dense(8, activation='relu')) # hidden layer with 32 units
model.add(Dense(4, activation='relu')) # hidden layer with 32 units
model.add(Dense(102, activation='sigmoid')) # output layer with 1 unit for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# evaluate the model on the testing data
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
losses.append(loss)
accuracies.append(accuracy)


# Plot the results
plt.plot(archs, losses)
plt.xlabel('Architecture')
plt.ylabel('Test loss')
plt.show()

plt.plot(archs, accuracies)
plt.xlabel('Architecture')
plt.ylabel('Accuracy')
plt.show()

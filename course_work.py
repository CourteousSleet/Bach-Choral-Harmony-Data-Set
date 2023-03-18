import numpy as np
import tensorflow as tf
from tensorflow import keras


# define function to read .data file
def read_data_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # split line into attributes
            attrs = line.strip().split(',')

            # convert pitch class attributes to binary (0 or 1)
            pitch_classes = [1 if x == 'YES' else 0 for x in attrs[2:14]]

            # convert meter to integer
            meter = int(attrs[15])

            # add row to data
            data.append([attrs[0], int(attrs[1])] + pitch_classes + [attrs[14], meter, attrs[16]])

    return np.array(data)


# set seed for reproducibility
np.random.seed(123)

# read data file
data = read_data_file('bach_chorales_harmony.data')

# split data into inputs (X) and labels (y)
X = data[:, 2:16].astype(int)
y = data[:, 16]

# split data into training and testing sets (80% for training, 20% for testing)
split = int(0.8 * len(data))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# define the neural network architecture
model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(len(harmony_distribution),), activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(harmony_distribution), activation='softmax')
])

# compile the model with appropriate loss function and optimizer
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# preprocess the data for training and testing
# assuming 'X_train', 'y_train', 'X_test', 'y_test' are already defined
X_train_preprocessed = tf.keras.utils.normalize(X_train, axis=1)
X_test_preprocessed = tf.keras.utils.normalize(X_test, axis=1)
y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=len(harmony_distribution))
y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=len(harmony_distribution))

# train the model
model.fit(X_train_preprocessed, y_train_categorical, epochs=10)

# evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test_preprocessed, y_test_categorical)
print('Test accuracy:', test_acc)
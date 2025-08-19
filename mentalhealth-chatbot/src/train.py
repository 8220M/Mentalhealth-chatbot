import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf

# Dummy dataset for now
texts = ['I feel sad', 'I am happy', 'I am anxious', 'Life is good']
labels = ['sad', 'happy', 'anxious', 'happy']

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=10)

# Label encoding
label_to_index = {l:i for i,l in enumerate(set(labels))}
y = np.array([label_to_index[l] for l in labels])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Simple LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=16, input_length=10),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(len(label_to_index), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, verbose=1)

# Save model
model.save('../model/chatbot_lstm.h5')

print('Training complete. Model saved!')

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 09:21:29 2022

@author: Sarah
"""

"""
Learning goals:
    (1) Prepare data and NN model is such a way that it will prone to overfit.
        Methods:
            - One hot encoding all categorical columns to increase the number of feature columns.
            - Increase model complexity (more dense layer, more nodes in each layers)
            - Train with high epochs
"""

from sklearn.model_selection import train_test_split
from sklearn import preprocessing, impute
import pandas as pd 
import numpy as np
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt

# STEP 1: Load Data
df = pd.read_csv(r"C:\Users\Sarah\Desktop\feedforward\data\german_credit\germanCredit.csv", delimiter=' ', header=None)

#%%
# STEP 2: Prepare Data
# Transform label column with values of 0 and 1 (currently it is in values of 1 and 2)
df[20] = df[20] - 1

# Extract label and features
label = df[[20]]
features = df.drop(columns=[20])

# Get dummies (one hot encoding) for each categorical columns
features = pd.get_dummies(features)

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.2, random_state = 0)
X_train = np.array(X_train)
X_test = np.array(X_test)

# Standardize features for train and test sets
standardizer = preprocessing.StandardScaler()
standardizer.fit(X_train)
X_train = standardizer.transform(X_train)
X_test = standardizer.transform(X_test)
#%%
# STEP 3: Prepare Model

# Get total number of categorical class to predict for
total_class = len(np.unique(np.array(y_train)))

inputs = tf.keras.Input(shape=(X_train.shape[-1],))
dense = tf.keras.layers.Dense(64, activation='relu')
x = dense(inputs)
dense = tf.keras.layers.Dense(32, activation='relu')
x = dense(x)
dense = tf.keras.layers.Dense(16, activation='relu')
x = dense(x)
dense = tf.keras.layers.Dense(8, activation='relu')
x = dense(x)
outputs = tf.keras.layers.Dense(total_class, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='german_credit_model_overfit')

#%%
# Compile and train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=16, epochs=70)

#%%
print(f'Mean Training Loss: {np.mean(history.history["loss"])}')
print(f'Mean Validation Loss: {np.mean(history.history["val_loss"])}')
print(f'Mean Training Accuracy: {np.mean(history.history["accuracy"])}')
print(f'Mean Validation Accuracy: {np.mean(history.history["val_accuracy"])}')

#%%
plt.plot(history.epoch, history.history['loss'], label='Training Loss')
plt.plot(history.epoch, history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.figure()

plt.plot(history.epoch, history.history['accuracy'], label='Training Accuracy')
plt.plot(history.epoch, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.figure()


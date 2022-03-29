# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:35:16 2022

@author: Sarah
"""

from sklearn.model_selection import train_test_split
from sklearn import preprocessing, impute
import pandas as pd 
import numpy as np
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt

# STEP 1: Load Dataset
df = pd.read_csv("https://raw.githubusercontent.com/SarahHannes/dl/main/feedforward/data/women_chess/top_women_chess_players_aug_2020.csv")

#%%
# STEP 2: Prepare Dataset
# Remove Fide id, Name, Gender
df = df.drop(columns=['Fide id', 'Name', 'Gender'])

# Encode inactive players to 1, otherwise 0
df['Inactive_flag'] = df['Inactive_flag'].apply(lambda x: 1 if x=='wi' else 0)

# Encode categorical value for Federation using Ordinal Encoder
enc = preprocessing.LabelEncoder()
enc.fit(df[['Federation']])
df['Federation'] = enc.transform(df[['Federation']])

# Drop blank Year_of_birth with median Year_of_birth of all players
df = df.dropna(subset='Year_of_birth')

# Fill blank Title with 'others'
df['Title'] = df['Title'].fillna(value='others')

# Fill blank ratings with 0
df['Rapid_rating'] = df['Rapid_rating'].fillna(value=0)
df['Blitz_rating'] = df['Blitz_rating'].fillna(value=0)

# One hot encoding for Title
df_cleaned = pd.get_dummies(df)
#%% 
# Identify X and y
X = df_cleaned.drop(columns=['Inactive_flag'])
y = df_cleaned[['Inactive_flag']]

# Split train/ test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
total_class = len(np.unique(y_test))

# Standardize
standardizer = preprocessing.StandardScaler()
standardizer.fit(X_train)
X_train = standardizer.transform(X_train)
X_test = standardizer.transform(X_test)

#%%
# STEP 3: Prepare Model

inputs = tf.keras.Input(shape=(X_train.shape[-1],))
dense = tf.keras.layers.Dense(16, activation='relu')
x = dense(inputs)
outputs = tf.keras.layers.Dense(total_class, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='women_chess_model')

#%%
# Compile and train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=8, epochs=25)

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

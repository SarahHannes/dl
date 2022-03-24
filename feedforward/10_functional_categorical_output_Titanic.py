# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 17:47:28 2022

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
train = pd.read_csv(r"C:\Users\Sarah\90min\Python\AI05\dl\datasets\titanic\train.csv")
test = pd.read_csv(r"C:\Users\Sarah\90min\Python\AI05\dl\datasets\titanic\test.csv") 

#%%
# STEP 2: Prepare Dataset
print('Before preprocessing:')
print('Train Set:')
print('null rows', train.isnull().sum())
print('na rows', train.isna().sum())

print('\nTest Set:')
print('null rows', test.isnull().sum())
print('na rows', test.isna().sum())
#%%
# Drop unused column
train = train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
test = test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

#%%
# Freq imputer for Embarked column
freq_imputer = impute.SimpleImputer(missing_values=np.nan, strategy='most_frequent')
freq_imputer.fit(train[['Embarked']])
train['Embarked'] = freq_imputer.transform(train[['Embarked']])

#%%
# Mean imputer for Age, Fare column
mean_imputer = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
mean_imputer.fit(train[['Age']])
train['Age'] = mean_imputer.transform(train[['Age']])

mean_imputer2 = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
mean_imputer2.fit(test[['Age']])
test['Age'] = mean_imputer2.transform(test[['Age']])

mean_imputer3 = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
mean_imputer3.fit(test[(test['Age']>=60) | (test['Age']<70)][['Fare']])
test['Fare'] = mean_imputer3.transform(test[['Fare']])


#%%
# One hot encoding for Sex, Embarked columns
train = pd.get_dummies(train)
test = pd.get_dummies(test)

#%%
print('\nAfter preprocessing:')
print('Train Set:')
print('null rows', train.isnull().sum())
print('na rows', train.isna().sum())

print('\nTest Set:')
print('null rows', test.isnull().sum())
print('na rows', test.isna().sum())

#%%
# Split features and labels
feature = train.drop(columns = ['Survived'])
label = train[['Survived']]

#%%
# Split into train and validation
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=1)

#%%
# Transform train and test sets using StandardScaler
standardizer = sklearn.preprocessing.StandardScaler()
standardizer.fit(X_train)
X_train = standardizer.transform(X_train)

standardizer2 = sklearn.preprocessing.StandardScaler()
standardizer2.fit(X_test)
X_test = standardizer2.transform(X_test)

#%%
# STEP 3: Prepare Model
inputs = tf.keras.Input(shape=(X_train.shape[-1],))
# dense = tf.keras.layers.Dense(32, activation='relu')
# x = dense(inputs)
dense = tf.keras.layers.Dense(16, activation='relu')
x = dense(inputs)
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='titanic_model')

#%%
# Compile and train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=20)

#%%
print(f'Mean Training Loss: {np.mean(history.history["loss"])}')
print(f'Mean Validation Loss: {np.mean(history.history["val_loss"])}')
print(f'Mean Training Accuracy: {np.mean(history.history["accuracy"])}')
print(f'Mean Validation Accuracy: {np.mean(history.history["val_accuracy"])}')

"""
Mean Training Loss: 0.4144014224410057
Mean Validation Loss: 0.47679068595170976
Mean Training Accuracy: 0.8278792113065719
Mean Validation Accuracy: 0.7913407742977142
"""
#%%
# Predict on out of sample test 
y_pred = np.argmax(model.predict(test), axis=1)
test['y_pred'] = pd.DataFrame(y_pred)

#%%
plt.plot(history.epoch, history.history['loss'], label='Training Loss')
plt.plot(history.epoch, history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.figure()

plt.plot(history.epoch, history.history['accuracy'], label='Training Accuracy')
plt.plot(history.epoch, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.figure()
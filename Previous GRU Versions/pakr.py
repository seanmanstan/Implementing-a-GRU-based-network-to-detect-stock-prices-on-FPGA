#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

mnist = tf.keras.datasets.mnist # 28x28 images of hand written digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)  #scaling data between 0 to 1, this is not necessary but makes training easier
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) #

model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu ))  #128 neurons and using relu = rectified linear layer 1 tf.nn.relu
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))  # layer 2
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))  #ouput layer, should have same neurons as possible outcome
#the activation is soft max cause outcome is a probability distribution
    
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#optimizer adjusts the gradient descent adam is the basic one

model.fit(x_train, y_train, epochs=3)


# In[2]:


val_loss, val_accuracy = model.evaluate(x_test, y_test)
print(val_loss, val_accuracy)

import matplotlib.pyplot as plt

#print(x_train[0])


# In[3]:


#model.save('name.model') #saves the model

#model = tf.keras.models.load_model('name.model') #loads saved model

#predictions = model.predict([x_test])  #this always takes a list
#the predictions are returned as probability distribution 

#import numpy as np
# print(np.argmax(predictions[0])) # display the prediction!


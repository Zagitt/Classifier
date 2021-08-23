import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ride.unhappened_ride.utils import generate_model

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

print(tf.__version__)
cols = ['id', 'completion', 'is_cc_ride', 'towards_distance', 'get_to_dropoff', 'get_to_pickup',
        'pickup_lapse', 'driver_called', 'passenger_called',
        'has_spike', 'linear_distance', 'service', 'is_approached', 'output']

input = pd.read_csv('model_generation_input.csv', names=cols, header=None)
# Drop rows with any empty cells
input = input.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
# service = pd.get_dummies(input.service, prefix='service')
input = input.drop(['service', 'id'], axis=1)
X = input.copy()
# X = pd.concat([service,input],axis=1)
y = X['output'].values
X = np.asarray(X).astype(np.float32)
sc = StandardScaler()

X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = generate_model(shape=X.shape[1])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=10e-5, clipvalue=0.5),
              metrics=['accuracy', 'mae', 'mean_squared_error'])

print(model.summary())

history = model.fit(X_train, y_train, batch_size=8, epochs=10, verbose=1, validation_split=0.2)
model.save_weights('/Users/giuseppemarotta/Documents/rpa-project/ride/unhappened_ride/model_generation/model_weights/',
                   save_format='tf')
print(input)

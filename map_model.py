import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.layers import Dense
from keras.models import Sequential
from keras import regularizers
import keras.backend as K

data = pd.read_csv("hourly_wages.csv")

train = data[['education_yrs', 'experience_yrs', 'female']].values
target = data[['wage_per_hour']].values

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.1, random_state = 2)

ranger_a = [35, 45]
ranger_b = [20, 30]
structure = {}
for a in range(ranger_a[0], ranger_a[1]+1):
    for b in range(ranger_b[0], ranger_b[1]+1):
        model = Sequential()
        model.add(Dense(a, activation = 'relu', input_shape = (train.shape[1],)))
        model.add(Dense(b, activation = 'relu'))
        model.add(Dense(1))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        model_training = model.fit(x = X_train, y = y_train, epochs = 500, validation_data = (X_test, y_test))
        structure[str(a) + '_' + str(b)] = [model_training.history['loss'][-1], model_training.history['val_loss'][-1]]
        print(f'Just tested: {a}_{b}', end = '\r')

import json

name = 'structure_ranger_' + str(ranger_a[0]) + '-' + str(ranger_a[1]) + \
       '_' + str(ranger_b[0]) + '-' + str(ranger_b[1]) + '.json'

with open(name, 'w') as f:
        json.dump(structure, f)

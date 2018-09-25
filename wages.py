import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.layers import Dense
from keras.models import Sequential

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Opening the Data
data = pd.read_csv("hourly_wages.csv")

train = data[['education_yrs', 'experience_yrs', 'female']].values
target = data[['wage_per_hour']].values

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.1, random_state = 2)

model = Sequential()
model.add(Dense(43, activation = 'relu', input_shape = (train.shape[1],)))
model.add(Dense(28, activation = 'relu'))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model_training = model.fit(x = X_train, y = y_train, epochs = 500, validation_data = (X_test, y_test))

# Linear Regressor

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
mean_squared_error(y_test, y_pred)

# Plotting

# Plotting the Data

x_data = np.empty((0,0))
y_data = np.empty((0,0))
z_data = np.empty((0,0))
for i in range(0, train.shape[0]):
    x_data = np.append(x_data, train[i,0])
    y_data = np.append(y_data, train[i,1])
    z_data = np.append(z_data, target[i])

# Plotting the Model
# Male
top = 25
predictions_m = np.empty((0,0))
predictions_f = np.empty((0,0))
pred_lr = np.empty((0,0))
x = np.empty((0,0))
y = np.empty((0,0))
for i in range(0, top + 1):
    for j in range(0, top + 1):
        x = np.append(x, i)
        y = np.append(y, j)
        predictions_m = np.append(predictions_m, model.predict(np.array([[i,j,0]])))
        predictions_f = np.append(predictions_f, model.predict(np.array([[i,j,1]])))
        pred_lr = np.append(pred_lr, regr.predict(np.array([[i,j,1]])))

fig1 = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter(x, y, predictions_m, label = 'Keras Male')
ax.scatter(x, y, predictions_f, label = 'Keras Female')
ax.scatter(x, y, pred_lr, label = 'Linear Regression')
ax.scatter(x_data, y_data, z_data, label = 'Data')
ax.set_xlabel('Education Years')
ax.set_ylabel('Experience Years')
ax.set_zlabel('Wage per Hour')
ax.set_title('3D Plot of the Models and the Data')
ax.legend()
plt.savefig('3D_Wages.jpeg')
plt.show()
plt.close()

# Is male = female?

x_data_m, x_data_f = np.empty((0,0)), np.empty((0,0))
y_data_m, y_data_f = np.empty((0,0)), np.empty((0,0))
z_data_m, z_data_f = np.empty((0,0)), np.empty((0,0))
for i in range(0, train.shape[0]):
    if train[i,2] == 0:
        x_data_m = np.append(x_data_m, train[i,0])
        y_data_m = np.append(y_data_m, train[i,1])
        z_data_m = np.append(z_data_m, target[i])
    else:
        x_data_f = np.append(x_data_f, train[i,0])
        y_data_f = np.append(y_data_f, train[i,1])
        z_data_f = np.append(z_data_f, target[i])

fig2 = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter(x_data_m, y_data_m, z_data_m, label = 'Male')
ax.scatter(x_data_f, y_data_f, z_data_f, label = 'Female')
ax.set_xlabel('Education Years')
ax.set_ylabel('Experience Years')
ax.set_zlabel('Wage per Hour')
ax.set_title('Data with Male vs Female Comparison')
ax.legend()
plt.show()
plt.close()

# Fixed Education Years

educ = 15
x = np.empty((0,0))
pred_lr = np.empty((0,0))
predictions_m = np.empty((0,0))
for j in range(0, top + 1):
    x = np.append(x, j)
    pred_lr = np.append(pred_lr, regr.predict(np.array([[educ,j,0]])))
    predictions_m = np.append(predictions_m, model.predict(np.array([[educ,j,0]])))

fig3 = plt.figure()
ax = plt.axes()
ax.plot(x, pred_lr, label = 'Linear Regression')
ax.plot(x, predictions_m, label = 'Keras Model')
ax.set_xlabel('Experience Years')
ax.set_ylabel('Wage per Hour')
ax.set_title(f'Wage per Hour x Experience Years for Males, given {educ} of Education Years')
ax.legend()
plt.show()
plt.close()

# Fixed Experience Years

exp = 10
x = np.empty((0,0))
pred_lr = np.empty((0,0))
predictions_m = np.empty((0,0))
for j in range(0, top + 1):
    x = np.append(x, j)
    pred_lr = np.append(pred_lr, regr.predict(np.array([[j,exp,0]])))
    predictions_m = np.append(predictions_m, model.predict(np.array([[j,exp,0]])))

fig4 = plt.figure()
ax = plt.axes()
ax.plot(x, pred_lr, label = 'Linear Regression')
ax.plot(x, predictions_m, label = 'Keras Model')
ax.set_xlabel('Education Years')
ax.set_ylabel('Wage per Hour')
ax.set_title(f'Wage per Hour x Education Years for Males, given {exp} of Experience Years')

ax.legend()
plt.show()
plt.close()

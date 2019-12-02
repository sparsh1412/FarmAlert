import keras
import pandas as pd
import numpy as np
import sklearn
import random

data = pd.read_csv(r'../dataset/Crop Yield/Final Dataset/Sugarcane.csv')
random.seed(0)

X = data.iloc[:,:486]
Y = data.iloc[:,486]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)

from keras.models import Sequential
from keras.layers import Dense	

def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(units=256, input_dim=486))
    regressor.add(Dense(units=256))
    regressor.add(Dense(units=128))
    regressor.add(Dense(units=1))
    adm = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    regressor.compile(optimizer=adm, loss='mean_squared_error',  metrics=['mse','accuracy'])
    return regressor

from keras.wrappers.scikit_learn import KerasRegressor
regressor = KerasRegressor(build_fn=build_regressor,epochs=10000)

results=regressor.fit(X_train,Y_train)

y_pred= regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
a = mean_squared_error(y_pred,Y_test)
print('MSE:', a)

regressor.model.save("../Kerasmodels/Sugarcanemodelkeras.h5")





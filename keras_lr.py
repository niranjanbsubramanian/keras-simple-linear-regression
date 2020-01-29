import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

df = pd.read_csv('Salary.csv')

X = df.iloc[:,0]
y = df.iloc[:,1]

model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Activation('linear'))

sgd = SGD(0.01)
model.compile(loss='mse',optimizer=sgd)

history = model.fit(X,y,epochs=500,verbose=0)

pred = model.predict(X)

plt.scatter(X, y, c='blue') 
plt.plot(X, pred, color='g') 
plt.show()

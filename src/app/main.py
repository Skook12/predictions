import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense

dataset = pd.read_csv('all_stocks_5yr.csv')
meio = len(dataset) // 2
meio = round(meio)
# Dividir o array pela metade
dataset_train = dataset[:meio]
dataset_test = dataset[meio:]

#Mostrar Conteudo dos Arrays
dataset_train.head()
dataset_test.head()

#Pre processamento de treinamento
train = dataset_train.loc[:, ['open']].values 
train


scaler = MinMaxScaler(feature_range = (0, 1))
train_scaled = scaler.fit_transform(train)
train_scaled

plt.plot(train_scaled)

X_train = []
y_train = []
timesteps = 50

for i in range(timesteps, 1250):
    X_train.append(train_scaled[i - timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Pre processamento de teste
test = dataset_test.loc[:, ['open']].values #array'e çevirdik
test

scaler = MinMaxScaler(feature_range = (0, 1))
test_scaled = scaler.fit_transform(test)
test_scaled

plt.plot(test_scaled)

X_test = []
y_test = []
timesteps = 50

for i in range(timesteps, 1250):
    X_test.append(test_scaled[i - timesteps:i, 0])
    y_test.append(test_scaled[i, 0])
    
X_test, y_test = np.array(X_test), np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Modelos

# Modelo de RNN Simples
model_rnn = Sequential([
    SimpleRNN(64, input_shape=(50, 1)),
    Dense(1)
])

model_rnn.compile(optimizer='adam', loss='mse')
model_rnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
loss_rnn = model_rnn.evaluate(X_test, y_test)

# Modelo LSTM
model_lstm = Sequential([
    LSTM(64, input_shape=(timesteps, 1)),
    Dense(1)
])

model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
loss_lstm = model_lstm.evaluate(X_test, y_test)

# Modelo GRU
model_gru = Sequential([
    GRU(64, input_shape=(timesteps, 1)),
    Dense(1)
])

model_gru.compile(optimizer='adam', loss='mse')
model_gru.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
loss_gru = model_gru.evaluate(X_test, y_test)

#Comparações entre os Modelos
print("Loss RNN:", loss_rnn)
print("Loss LSTM:", loss_lstm)
print("Loss GRU:", loss_gru)

def plot_predictions(model, X_test, Y_test):
    predictions = model.predict(X_test)
    plt.plot(Y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.show()
print("RNN")
plot_predictions(model_rnn, X_test, y_test)
print("LSTM")
plot_predictions(model_lstm, X_test, y_test)
print("GRU")
plot_predictions(model_gru, X_test, y_test)
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))
model.compile(optimizer='rmsprop', loss='mse')

model.fit(x_data, y_data, nb_epoch=200, batch_size=32)

print("Done")
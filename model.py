import keras
import keras.layers as kl
from keras.models import Sequential

length = 441
location = 'model.h5'

def makemodel():
    model = Sequential()

    model.add(kl.Conv1D(4,
                        kernel_size=3,
                        padding='same',
                        input_shape=(length, 2),
                        activation='relu'))

    model.add(kl.MaxPool1D(pool_size=2))

    model.add(kl.Conv1D(128, 3,
                        strides=1,
                        padding='same',
                        activation='relu'))

    model.add(kl.MaxPool1D(pool_size=2))

    model.add(kl.Conv1D(128, 3,
                        strides=1,
                        padding='same',
                        activation='relu'))

    model.add(kl.MaxPool1D(pool_size=2))

    model.add(kl.Flatten())

    model.add(kl.Dense(units=length,
                       activation='relu'))

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

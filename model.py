import keras
import keras.layers as kl
from keras.models import Sequential

length = 2500
location = 'model_wide.h5'

def makemodel():
    model = Sequential()

    model.add(kl.Conv1D(128,
                        kernel_size=16,
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

    model.add(kl.AvgPool1D(pool_size=16))

    model.add(kl.Flatten())

    model.add(kl.Dense(units=length,
                       activation='sigmoid'))

    model.compile(loss=keras.losses.mean_absolute_percentage_error,
                  optimizer=keras.optimizers.Adagrad(),
                  metrics=['accuracy'])
    return model

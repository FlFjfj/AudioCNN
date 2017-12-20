import model
import numpy as np
import keras

def main():
    mod = keras.models.load_model(model.location)
    data = [[i / model.length] * 2 for i in range(model.length)]
    pred = mod.predict(np.asarray([data]))
    print(pred)


if __name__ == "__main__":
    main()

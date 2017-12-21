import struct

import model
import numpy as np
import keras
import datasetLoader
import wave


def main():
    data = datasetLoader.loadTestset("data.wav")
    out = wave.open("out.wav", "w")
    out.setframerate(44100)
    out.setsampwidth(1)
    out.setnchannels(1)

    mod = keras.models.load_model(model.location)
    res = mod.predict(np.asarray(data))

    for r in res:
        print(np.sum(np.abs(r)))
        for b in r:
            out.writeframes(struct.pack('B', int(b * 255)))
    out.close()


if __name__ == "__main__":
    main()

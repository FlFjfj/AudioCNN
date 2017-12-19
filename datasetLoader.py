import wave

import model

def loadDataset(path):
    file = wave.open(path)
    print(file.getnframes())

    input_data = []
    predict = []

    left = file.getnframes()
    while left >= model.length + model.predict:
        input_data.append(file.readframes(model.length))
        predict.append(file.readframes(model.predict))
        left -= model.length + model.predict

    return input_data, predict

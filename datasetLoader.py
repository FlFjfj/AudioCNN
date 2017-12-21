import wave
import struct
import numpy as np
import model

step = int(model.length / 8)
set_size = 8000


def loadDataset(path):
    file = wave.open(path)

    input_data = []
    predict = []

    file_buffer = file.readframes(model.length + step * (set_size - 1))
    for i in range(set_size):
        buf = file_buffer[i * step:i * step + model.length]
        buf = list(map(lambda x: x / 255, buf))

        input_data.append([[i, i] for i in buf])
        predict.append(buf)

    print("Dataset loaded.")
    return input_data, predict


def loadTestset(path):
    file = wave.open(path)

    input_data = []

    parts = int((model.length + step * (set_size - 1)) / model.length) + 1
    file_buffer = file.readframes(parts * model.length)
    for i in range(parts):
        buf = file_buffer[i * model.length:(i+1) * model.length]
        buf = list(map(lambda x: x / 255, buf))

        input_data.append([[i, i] for i in buf])

    print("Testset loaded.")
    return input_data


def loadData(patha, pathb):
    filea = wave.open(patha)
    fileb = wave.open(pathb)

    input_data = []

    parts = int((model.length + step * (set_size - 1)) / model.length) + 1
    file_buffera = filea.readframes(parts * model.length)
    file_bufferb = fileb.readframes(parts * model.length)

    for i in range(parts):
        bufa = file_buffera[i * model.length:(i + 1) * model.length]
        bufb = file_bufferb[i * model.length:(i + 1) * model.length]

        bufa = list(map(lambda x: x / 255, bufa))
        bufb = list(map(lambda x: x / 255, bufb))

        input_data.append([[bufa[i], bufb[i]] for i in range(model.length)])

    print("Data loaded.")
    return input_data
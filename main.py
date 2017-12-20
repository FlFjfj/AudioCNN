import numpy as np
import tensorflow as tf

import model
import datasetLoader
import wave

def main():
    input_layer, result_layer, predict_layer, loss = model.makemodel()

    input_data = [[[(i / model.length + j / model.batch) / 2] for i in range(model.length)]
                  for j in range(model.batch)]
    prediction_data = [[(i / model.length + j / model.batch) / 2 for i in range(model.predict)]
                       for j in range(model.batch)]

    optimizer = tf.train.AdamOptimizer(0.0001)
    train = optimizer.minimize(loss, name="optimizer")
    writer = tf.summary.FileWriter("./", graph=tf.get_default_graph())
    tf.summary.scalar("loss", loss)
    summary = tf.summary.merge_all()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    meta_data = tf.RunMetadata()

    saver = tf.train.Saver();
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    audio_data, pred_data = datasetLoader.loadDataset('techicolorbeat.wav')
    print(len(audio_data), model.length)
    with tf.name_scope("train"):
        leftL = 0
        leftP = 0
        for i in range(len(audio_data)):
            input_data = [[[s] for s in audio_data[i][:model.length]]]
            prediction_data = [[p for p in audio_data[i][:model.predict]]]
            data, lossval, _ = session.run([summary, loss, train],
                                            {input_layer: input_data, predict_layer: prediction_data},
                                            options=run_options,
                                            run_metadata=meta_data)
            writer.add_run_metadata(meta_data, 'step%d' % i)
            writer.add_summary(data, i)
            print("Written data for ", i, ", loss=", lossval)
            for j in range(5):
                session.run(train, {input_layer: input_data, predict_layer: prediction_data})
    #saver.save(session, "./model.cpkt")
    sound_data = np.array(1)
    for i in range(100):
        result = session.run(result_layer, {input_layer: input_data})
        new_data = [[i] for i in result[0]]
        input_data = input_data[0][len(new_data):]
        input_data.extend(new_data)
        input_data = [input_data]
        sound_data = np.append(sound_data, [result[0]])
        print("written samples " + str(i))
    a = wave.open("./audio.wav", 'w')
    a.setsampwidth(4)
    a.setframerate(44100);
    a.setnframes(len(sound_data))
    a.setnchannels(1)
    a.writeframes(sound_data)
    a.close()
    print("audio written");
if __name__ == "__main__":
    main()

import numpy as np
import tensorflow as tf

import model
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

    with tf.name_scope("train"):
        for i in range(20):
            if i % 5 == 0:
                data, lossval, _ = session.run([summary, loss, train],
                                               {input_layer: input_data, predict_layer: prediction_data},
                                               options=run_options,
                                               run_metadata=meta_data)
                writer.add_run_metadata(meta_data, 'step%d' % i)
                writer.add_summary(data, i)
                print("Writed data for ", i, ", loss=", lossval)
            else:
                session.run(train, {input_layer: input_data, predict_layer: prediction_data})
    #saver.save(session, "./model.cpkt")

    result = session.run(result_layer, {input_layer: input_data})
    a = wave.open("./audio.wave", 'w')
    a.setsampwidth(4)
    a.setframerate(4410);
    a.setnframes(len(result[0]))
    a.setnchannels(1)
    a.writeframes(result[0])
    a.close()
    print("result: ", result)
if __name__ == "__main__":
    main()

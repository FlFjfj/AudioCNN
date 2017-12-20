import model
import numpy as np

def main():
    np.random.seed(7)

    mod = model.makemodel()

    data = [[[i / model.length] * 2 for i in range(model.length)] for _ in range(100)]
    result = [[i / model.length for i in range(model.length)] for _ in range(100)]

    mod.fit(x=data,
            y=result,
            batch_size=2,
            epochs=4)

    model.save('model.h5')

if __name__ == "__main__":
    main()

import model
import datasetLoader

def main():
    data, result = datasetLoader.loadDataset("data.wav")

    mod = model.makemodel()
    mod.fit(x=data,
            y=result,
            batch_size=8,
            epochs=16,
            shuffle=True)

    mod.save(model.location)


if __name__ == "__main__":
    main()

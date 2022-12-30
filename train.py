import tensorflow


class Train:

    def build(self):
        mnist = tensorflow.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = tensorflow.keras.utils.normalize(x_train, axis=1)
        x_test = tensorflow.keras.utils.normalize(x_test, axis=1)

        nn_model = tensorflow.keras.models.Sequential()
        nn_model.add(tensorflow.keras.layers.Flatten(input_shape=(28, 28)))
        nn_model.add(tensorflow.keras.layers.Dense(128, activation="relu"))
        nn_model.add(tensorflow.keras.layers.Dense(128, activation="relu"))
        nn_model.add(tensorflow.keras.layers.Dense(10, activation="softmax"))
        nn_model.compile(
            metrics=["accuracy"],
            loss="sparse_categorical_crossentropy",
            optimizer="adam")

        nn_model.fit(x_train, y_train, epochs=5)
        nn_model.save("nn.model")

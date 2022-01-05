import tensorflow as tf
import tensorflow.keras as keras

class ConvNet(keras.Model):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = keras.layers.Conv2D(32, 3) # into 32 channels, kernel size: 3
        self.flatten = keras.layers.Flatten()
        self.d1 = keras.layers.Dense(128)
        self.d2 = keras.layers.Dense(10)
        self.relu = keras.layers.ReLU()
        self.softmax = keras.layers.Softmax(axis = -1)

    def call(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = self.d1(x)
        x = self.relu(x)
        x = self.d2(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':

    mnist = keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    # Batch and shuffle
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    model = ConvNet()

    loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits = False) # criterion # from_logits is false if passed through a softmax
    optimizer = keras.optimizers.Adam()

    train_loss = keras.metrics.Mean(name = 'train_loss')
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

    test_loss = keras.metrics.Mean(name = 'test_loss')
    test_accuracy = keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 5
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
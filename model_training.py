import os
import tensorflow as tf
import numpy as np

def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def preprocess_data(x, y, num_classes=10):
    x = x.astype("float32") / 255
    x = np.expand_dims(x, -1)
    y = tf.keras.utils.to_categorical(y, num_classes)
    return x, y

def load_or_train_model(x_train, y_train, model_path, num_classes=10):
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Training new model...")
        input_shape = (28, 28, 1)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10, validation_split=0.2)
        model.save(model_path)
        print("Model trained and saved at", model_path)
    return model

def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc*100:.2f}%")

def main():
    model_path = 'model.keras'  # Use .keras or .h5 extension
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = load_data()

    x_test, y_test = preprocess_data(x_test, y_test, num_classes)
    x_train, y_train = preprocess_data(x_train, y_train, num_classes)

    model = load_or_train_model(x_train, y_train, model_path, num_classes)
    evaluate_model(model, x_test, y_test)

if __name__ == "__main__":
    main()

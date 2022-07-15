import tensorflow as tf
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
input_shape = (28, 28, 1)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train=x_train / 255.0

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test=x_test/255.0
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)
batch_size = 64
num_classes = 10
epochs = 10
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dropout(0.25),
    # tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    )
model.save("full_model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)
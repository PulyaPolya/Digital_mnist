from keras.datasets import mnist
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
from matplotlib import pyplot
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()
from matplotlib import pyplot as plt
(x_train, y_train), (x_test, y_test) = mnist.load_data()
input_shape = (28, 28, 1)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train=x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test=x_test/255.0
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)
reconstructed_model =tf.keras.models.load_model("my_h5_model.h5")

test_loss, test_acc = reconstructed_model.evaluate(x_test, y_test)
batch_size = 64
num_classes = 10
epochs = 2
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    )
test_loss2, test_acc2 = model.evaluate(x_test, y_test)
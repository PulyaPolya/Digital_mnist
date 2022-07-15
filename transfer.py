import tensorflow as tf
from keras.datasets import mnist
import numpy as np

def expand(img):
    img = np.array(img)
    img = np.pad(img, pad_width=[(2, 2), (2, 2)], mode='constant')
    img = np.expand_dims(img, 0)
    img = img.repeat(3, axis =0)
    return img
(x_train, y_train), (x_test, y_test) = mnist.load_data()
input_shape = (28, 28, 1)

# x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
# x_train=x_train / 255.0
#
# # x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
# x_test=x_test/255.0
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)
x_train_3_dims = []
for i in range(x_train.shape[0]):
    new_x = expand(x_train[i])
    x_train_3_dims.append(new_x)
x_test_3_dims  = []
for i in range(x_test.shape[0]):
    new_x = expand(x_test[i])
    x_test_3_dims.append(new_x)
x_test_3_dims = np.array(x_test_3_dims)
IMG_SIZE = (32, 32)
IMG_SHAPE = IMG_SIZE + (3,)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(x_train)
inputs = tf.keras.Input(shape=(32, 32, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

initial_epochs = 10
validation_dataset = tf.data.Dataset.from_tensor_slices(x_test_3_dims,y_test )
loss0, accuracy0 = model.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
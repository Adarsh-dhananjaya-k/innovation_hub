# import cv2
# import numpy as np
# import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# # model = tf.keras.models.Sequential()

# # model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# # model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# # model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))


# # model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# # model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# # model.fit(x_train, y_train, epochs=3)
# # model.save('number.keras')

# model = tf.keras.models.load_model('number.keras')
# loss ,acc = model.evaluate(x_test, y_test)
# print(loss, acc)

import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=3)
# model.save('number.keras')

model = tf.keras.models.load_model('number.keras')
loss, acc = model.evaluate(x_test, y_test)
print(loss, acc)

img = cv2.imread('Untitled.png')[:,:,0]
img = np.invert(np.array([img]))
# plt.imshow('image', img[0],cmap=plt.cm.binary)
# plt.show()

prediction = model.predict(img)
print("the number is ",np.argmax(prediction))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

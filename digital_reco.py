import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model


#  Load the dataset 
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(y_test)

# Normalize the dataset
x_test = tf.keras.utils.normalize(x_test, axis=1)
x_train_normalized = tf.keras.utils.normalize(x_train,axis=1)

# plt.figure(figsize=(10, 4))

# # Before normalization

# for i in range(len(x_train_normalized)):
#     plt.subplot(1, 2, 1)
#     plt.title("Before Normalization")
#     plt.imshow(x_train[i], cmap='gray')

#     # After normalization
#     plt.subplot(1, 2, 2)
#     plt.title("After Normalization")
#     plt.imshow(x_train_normalized[i], cmap='gray')

#     plt.show()

# # Load the model
# model = tf.keras.models.Sequential()

# print(x_train_normalized[0].shape)

# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
# model.add(tf.keras.layers.Dense(128, activation="relu"))
# model.add(tf.keras.layers.Dense(10, activation="softmax"))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# # inseritng the data to the model for training
# model.fit(x_train_normalized,y_train, epochs=5)


# # testing the model
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print("Test Loss: ", test_loss)
# print("Test Accuracy: ", test_acc)

# model.save('digit_recognizer.keras')


model_loaded = tf.keras.models.load_model('digit_recognizer.keras')

test_img = cv2.imread('img1.png')[:,:,0]
# Display the image
print(test_img.shape)
plt.imshow(test_img, cmap=plt.cm.binary)
plt.show()
test_img = np.invert(np.array([test_img]))
prediction = model_loaded.predict(test_img)
print(prediction)
print("the number is ",np.argmax(prediction))
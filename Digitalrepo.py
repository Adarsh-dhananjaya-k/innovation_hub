import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import tensorflow as tf 
# from tensorflow.keras.models load model 

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(y_test)

x_test=tf.keras.utils.normalize(x_test,axis=1)
x_train_normalized=tf.keras.utils.normalize(x_train,axis=1)

plt.figure(figsize=(10,4))

model=tf.keras.models.Sequential()

print(x_train_normalized[0].shape)
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train_normalized,y_train,epochs=5)

test_loss,test_acc=model.evaluate(x_test,y_test)
print("test loss:",test_loss)
print("test accuracy:",test_acc)


test_img=cv2.imread('image1.png')[:,:,0]

test_img = cv2.resize(test_img,(28,28))
plt.imshow(test_img,cmap=plt.cm.binary)
plt.show()
test_img =np.invert(np.array([test_img]))
prediction=model.predict(test_img)
print(prediction)
print("the number is",np.argmax(prediction))


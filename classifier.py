import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten

#Load and Inspect the Data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print("Training set shape:", x_train.shape, y_train.shape)
print("Fashion MNIST loader docstring:\n", fashion_mnist.load_data.__doc__)

#Define Class Labels
class_names = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

#Visualize a Sample
index = 2
plt.imshow(x_train[index], cmap="gray")
plt.title(f"Label: {class_names[y_train[index]]}")
plt.show()

#Normalize the Pixel Values
x_train = x_train / 255.0
x_test = x_test / 255.0

#One-Hot Encode the Labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print("Test labels shape after one-hot encoding:", y_test.shape)

#Build the Neural Network
model = Sequential([
    Input((28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.summary()

#Compile the Model
model.compile(
    loss=tf.losses.CategoricalCrossentropy(),
    optimizer=tf.optimizers.Adam(),
    metrics=[tf.metrics.CategoricalAccuracy()]
)

#Train the Model
model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=50,
    verbose=2
)

#Evaluate the Model 
model.evaluate(x_test, y_test)

#Predict and Visualize a Sample
y_pred = model.predict(x_test)

index = 1
plt.imshow(x_test[index], cmap='gray')
plt.title("Sample from Test Set")
plt.show()

print(f'Predicted: {y_pred[index].argmax()}')
print(f'True: {y_test[index].argmax()}')

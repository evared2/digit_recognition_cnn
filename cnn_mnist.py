import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Loading the MNIST dataset (60k images for training, 10k images for testing)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Data preprocessing
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))  # 28x28 εικόνες με 1 κανάλι (άσπρο-μαύρο)
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Normalization of images (0-255 -> 0-1)
train_images, test_images = train_images / 255.0, test_images / 255.0

# Creating the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 κατηγορίες για τα ψηφία 0-9
])

# Summary description of the model
model.summary()

# Model compilation (defining optimizer, loss function, and metrics)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Model training
history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# Model evaluation with test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Saving the model for future use
model.save('mnist_cnn_model.h5')

# Visualization of the training history (accuracy and loss)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Model Accuracy')
plt.savefig('accuracy_plot.png')
plt.show()

# Predictions with the trained model
predictions = model.predict(test_images)

# Displaying example prediction
fig, axes = plt.subplots(1, 5, figsize=(10, 5))
for i in range(5):
    axes[i].imshow(test_images[i].reshape(28, 28), cmap='gray')
    axes[i].set_title(f"Pred: {np.argmax(predictions[i])}, True: {test_labels[i]}")
    axes[i].axis('off')

#Saving example prediction
plt.savefig('example_predictions.png')
plt.show()















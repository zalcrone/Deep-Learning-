# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset_2/seg_train',
                                                 target_size = (128, 128),
                                                 batch_size = 50,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset_2/seg_test',
                                            target_size = (128, 128),
                                            batch_size = 50,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=150, kernel_size=3, activation='relu', input_shape=[128, 128, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(4, 4))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=120, kernel_size=3, activation='relu', padding = 'same'))


cnn.add(tf.keras.layers.Conv2D(filters=80, kernel_size=3, activation='relu', padding = 'same'))


cnn.add(tf.keras.layers.MaxPool2D(4, 4))
# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=100, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=6, activation='softmax'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 32)

# Part 4 - Making a single prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset_2/seg_pred/1386.jpg', target_size = (128, 128, 3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
training_set.class_indices
result = cnn.predict(test_image)

if result[0][0] == 1:
    predictions = 'buildings'
elif result[0][1] == 1:
    predictions = 'forest'
elif result[0][2] == 1:
    predictions = 'glacier'
elif result[0][3] == 1:
    predictions = 'mountain'
elif result[0][4] == 1:
    predictions = 'sea'
else:
    predictions = 'street'
    
print(predictions)

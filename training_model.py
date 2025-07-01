from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf


# Define parameters
batch_size = 32
epochs = 30
num_classes = 18

# Define image data generator
# Define image data generator
data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,  # added data augmentation
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

# Define train and validation data generator
train_gen = data_gen.flow_from_directory(directory='/content/drive/MyDrive/Bima Project/Modig1/Ready',
                                         target_size=(224, 224),
                                         color_mode='rgb',
                                         batch_size=batch_size,
                                         class_mode='categorical',
                                         subset='training')

val_gen = data_gen.flow_from_directory(directory='/content/drive/MyDrive/Bima Project/Modig1/Ready',
                                       target_size=(224, 224),
                                       color_mode='rgb',
                                       batch_size=batch_size,
                                       class_mode='categorical',
                                       subset='validation')

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Fine-tune the model by unfreezing the last few layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Freeze base layers
#//for layer in base_model.layers:
#   layer.trainable = False//

# Add new layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='min')
checkpoint = ModelCheckpoint(filepath='best_model.h5', save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min')

# Train the model
model.fit(train_gen,
          steps_per_epoch=train_gen.samples // batch_size,
          validation_data=val_gen,
          validation_steps=val_gen.samples // batch_size,
          epochs=epochs,
          callbacks=[early_stop, checkpoint])

# Get the class labels and indices
class_labels = train_gen.class_indices

# Save the class labels and indices to a file
with open('labels.txt', 'w') as f:
    for label, index in class_labels.items():
        f.write(f'{label}, {index}\n')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('model5.tflite', 'wb') as f:
    f.write(tflite_model)

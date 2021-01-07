"""
Authors: Cas van Rijbroek & Lex Bosch
Last modified: 7 January 2021

This file contains the implementation of the CNN. Parameters are defined in the main function and the other scripts are
called for the collection of the data and the visualization of the results.
"""

# The following order of imports is necessary for the model to run on a plaidml backend
# Plaidml should be configured before running the script
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras

from data_gathering import get_generators

from keras import callbacks
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Flatten, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from classification_results import print_results


#def main():
"""The main function can be edited at will to change the parameters of the model or the data.
"""
train_generator, test_generator, validate_generator = get_generators(batch_size=32)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(256, 256, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(2, 2), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(4, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

early_stopping = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validate_generator.n//validate_generator.batch_size
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=test_generator,
                              validation_steps=STEP_SIZE_TEST,
                              callbacks=[early_stopping],
                              epochs=50)

print_results(model, test_generator, validate_generator)


#if __name__ == "__main__":
#    main()

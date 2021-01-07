"""
Authors: Cas van Rijbroek & Lex Bosch
Last modified: 7 January 2021

This file contains code to validate and visualize the results of the model.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report


def print_results(model, test_generator, validate_generator=None):
    """First a prediction is made using the test generator and the model. This is used to create a confusion matrix.
    Then a classification report is printed containing more performance measures (precision, recall, f1-score) for the
    different classes.

    If a final validation generator is provided, the function will recursively repeat the process on this generator.

    :param model: the Keras model
    :param test_generator: a keras generator for the test data
    :param validate_generator: (optional) a keras generator for the final validation data
    """
    if validate_generator is None:
        title = "test"
    else:
        title = "final validation"

    y_pred = model.predict_generator(test_generator)
    y_pred = np.argmax(y_pred, axis=1)
    df_cm = pd.DataFrame(confusion_matrix(test_generator.classes, y_pred, normalize="true"),
                         index=test_generator.class_indices.keys(), columns=test_generator.class_indices.keys())

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, fmt="g", annot=True, cbar=False, cmap=sn.dark_palette("#ffd000", reverse=True, as_cmap=True))
    plt.title(f"Confusion matrix of bee/wasp classification on {title} data")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    print('Classification Report')
    target_names = test_generator.class_indices.keys()
    print(classification_report(test_generator.classes, y_pred, target_names=target_names))

    if validate_generator is not None:
        print("Final validation")
        print("*" * 50)
        print_results(model, validate_generator)

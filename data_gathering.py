"""
Authors: Cas van Rijbroek & Lex Bosch
Last modified: 7 January 2021

This file contains the code needed to gather the data needed to run the model (pictures of bees, wasps, insects and
other).

The repository structure of your data is not important to the gathering process. A csv should be provided in the
kaggle_bee_vs_wasp repository called labels.csv. This file should contain the following columns for every image:
path, is_bee, is_wasp, is_otherinsect, is_other, photo_quality, is_validation, is_final_validation

Where the path indicates the relative file location, is_bee to is_other the class id (1 if true, else 0), photo quality
indicates whether it is a high quality photo (1) or not (0) and is_validation and is_final_validation if the data is part
of the test or validation set (1), else (0).
"""

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator


def get_generators(batch_size, filter_quality=False):
    """Creates keras generators for the train, test and validation sets.

    :param batch_size: the batch size that should be used for every step of every epoch of training the model
    :param filter_quality: set to True if you only want high quality images (this removes the 'other' class and thus
    makes it a 3 class dataset!)
    :return: tuple containing the train, test and validation generators (in that order)
    """
    df = pd.read_csv("kaggle_bee_vs_wasp\\labels.csv")

    update_dataframe(df)

    if filter_quality:
        df = df.loc[df["photo_quality"] == 1]

    train_df = df.loc[(df["is_validation"] == 0) & (df["is_final_validation"] == 0)]
    test_df = df.loc[df["is_validation"] == 1]
    validate_df = df.loc[df["is_final_validation"] == 1]

    train = ImageDataGenerator(rescale=1./255)
    test = ImageDataGenerator(rescale=1./255)
    validate = ImageDataGenerator(rescale=1./255)

    train_generator = train.flow_from_dataframe(
        dataframe=train_df,
        x_col="path",
        y_col="label",
        batch_size=batch_size,
        class_mod="binary",
        target_size=(256, 256),
        shuffle=False)
    test_generator = test.flow_from_dataframe(
        dataframe=test_df,
        x_col="path",
        y_col="label",
        batch_size=batch_size,
        class_mod="binary",
        target_size=(256, 256),
        shuffle=False)
    validate_generator = validate.flow_from_dataframe(
        dataframe=validate_df,
        x_col="path",
        y_col="label",
        batch_size=batch_size,
        class_mod="binary",
        target_size=(256, 256),
        shuffle=False)

    return train_generator, test_generator, validate_generator


def update_dataframe(df):
    """Adds a label column to the csv that contains the information from the 4 label columns in strings representing the
    4 classes. This is done, because Keras expects the class labels to be in a single column for a generator to be
    created.

    :param df: Pandas dataframe of the labels.csv file
    :return: the new dataframe containing the label column
    """
    for index, row in df.iterrows():
        df.loc[index, "path"] = f"kaggle_bee_vs_wasp\\{row['path']}"

        if row["is_bee"] == 1:
            df.loc[index, "label"] = "bee"
        elif row["is_wasp"] == 1:
            df.loc[index, "label"] = "wasp"
        elif row["is_otherinsect"] == 1:
            df.loc[index, "label"] = "insect"
        elif row["is_other"] == 1:
            df.loc[index, "label"] = "other"
        else:
            raise ValueError(f"Entry {index} contains no valid class label")

    return df

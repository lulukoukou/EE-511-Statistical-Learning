#!/usr/bin/env python2
"""Data interface help functions.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import csv


def load_csv(filename):
    """Loads CSV file.

    Args:
        filename: String for the CSV filename.

    Returns:
        lines: List of text lines.
    """
    lines = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            lines.append(line)
    return lines


def load_adult_data():
    """Loads the adult data.

    Returns:
        The full adult in the form of list of text lines.
    """
    return load_csv("adult_train.csv")


# Note: Possibly use different data for training and validation to get a more accurate result,
# but remember that in the last part your model will be trained on the full training data
# load_adult_data() and be tested on a test dataset you don't have access to.

##def load_adult_train_data_for_test_accuracy():
##    return load_adult_data()

def load_adult_train_data():
    return load_adult_data()
##    temp = load_adult_data()
##    train = []
##    for i in range(len(temp)):
##        if i % 10 != 2 and i % 12 != 5 and i % 5 != 4:
##            train.append(temp[i])
##    return train


def load_adult_valid_data():
    return load_adult_data()
##    temp = load_adult_data()
##    validation = []
##    for i in range(len(temp)):
##        if i % 10 == 2 or i % 12 == 5 or i % 6 == 4:
##            validation.append(temp[i])
##    return validation

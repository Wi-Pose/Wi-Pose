# from __future__ import division
import copy
import numpy as np

def MINMAXNormalization(matrix,):

    amin, amax = np.min(matrix), np.max(matrix)
    nor_a = (matrix - amin) / (amax - amin)
    return nor_a

def pictureNormalizationForRGB(rgbmaxtrix):
    RGB = None
    for matrix in rgbmaxtrix:
        raw = matrix / 255.
        if RGB is None:
            RGB = np.array([raw])
        else:
            RGB = np.append(RGB, [raw], axis=0)
        print RGB

    return RGB

def scalerNormalization(matrix):
    """"
    The Column Method
    """
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    nor_a=   min_max_scaler.fit_transform(matrix)
    return nor_a
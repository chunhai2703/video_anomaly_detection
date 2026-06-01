import numpy as np

def apply_threshold(scores, threshold):

    predictions = []

    for score in scores:

        if score > threshold:
            predictions.append(1)
        else:
            predictions.append(0)

    return np.array(predictions)
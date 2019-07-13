from typing import List, Tuple, Union

import numpy as np


def relu(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    ReLU function

    Parameters
    ----------
    x: float or array

    Returns
    -------
    the same type and the shape as the input

    """

    return np.maximum(x, 0.0)


def tanh(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    tanh function

    Parameters
    ----------
    x: float or array

    Returns
    -------
    the same type and the shape as the input

    """
    
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    sigmoid function

    Parameters
    ----------
    x : float or array

    Returns
    -------
    the same type and the shape as the input

    """

    return 1.0 / (1.0 + np.exp(-x))


def binary_cross_entropy_with_logit(logit: float, label: int) -> float:
    """
    Cross entropy loss for binary classification

    Parameters
    ----------
    logit : float
        sigmoid(s) is the predicted probability that the label is 1
    label : int
        the correct label 0 or 1

    Returns
    -------
    float

    """

    if label not in [0, 1]:
        raise ValueError("label should be 0 or 1")

    # avoid overflow
    if logit < 50.0:
        xent_loss = label * np.log(1+np.exp(-logit)) + (1-label) * np.log(1+np.exp(logit))
    else:
        xent_loss = label * np.log(1+np.exp(-logit)) + (1-label) * logit

    return xent_loss


def evaluation(model, graphs: List[np.ndarray], labels: List[int]) -> Tuple[float, float]:
    """
    Evaluate model with dataset

    Parameters
    ----------
    model : insatnce of GnnBinaryClassfiier
    graphs : list of array
    labels : list of int

    Returns
    -------
    accuracy : float
    average_loss : float

    """

    num_of_dataset = len(graphs)
    correct = 0
    loss = 0.0
    for graph, label in zip(graphs, labels):
        pred = model.prediction(graph)
        if pred == label:
            correct += 1
        loss += model.loss(graph, label)
    accuracy = float(correct) / num_of_dataset
    average_loss = loss / num_of_dataset
    return accuracy, average_loss

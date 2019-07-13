from typing import List

import numpy as np

from model.tools import Params
from model.funcs import relu, sigmoid, tanh


class GnnGraphFeature(object):
    """
    The class for the graph of GNN.
    This class reads a graph and performs aggregation and readout to generate the feature of the graph.

    """

    def __init__(self, params: Params) -> None:
        """
        Parameters
        ----------
        params : class Params

        """

        self.feature_dim = params.feature_dim  # size of graph feature
        self.feature = None  # graph feature which will be calculated by aggregation and readout

        layers = params.layers  # None or list of integer [input size, hidden unit 1, ... , hidden unit n, output size]
        if layers is None:  # if layers is Nine, one layer perception is used
            self.layers = [self.feature_dim, self.feature_dim]
        else:
            if not isinstance(layers, list):
                raise ValueError("layers should be list")
            if layers[0] != self.feature_dim or layers[-1] != self.feature_dim:
                raise ValueError("The input and output sizes should be the same as the feature size")

            self.layers = layers

        self.W = None  # weight for aggregation

        # activation function for aggregation
        if params.activation_for_graph_feature == "relu":
            self.activf = relu
        elif params.activation_for_graph_feature == "tanh":
            self.activf = tanh
        elif params.activation_for_graph_feature == "sigmoid":
            self.activf = sigmoid
        else:
            raise ValueError("The specified activation function is not defined")

    def __call__(self, graph: np.ndarray, num_of_steps=2) -> np.ndarray:
        """
        Performs aggregation and readout, then return the feature vector of the graph h_g.

        Parameters
        ----------
        graph : array
            The graph which will be classified.
            The graph is represented as a adjacent matrix, which is a square symmetric matrix.
        num_of_steps : int
            The number of steps for aggregation. The initial value is 2.

        Returns
        -------
        array
            Array with size (self.feature_dim). The result of aggregation and readout.

        """

        if self.W is None:
            raise ValueError("The weight (self.W) is not initialized")

        if np.shape(graph)[0] != np.shape(graph)[1]:
            raise ValueError("Invalid shape of graph. "
                             "The graph should be adjacent matrix, which is a square symmetric matrix.")

        feature = self.initialize_features(np.shape(graph)[0])

        # aggregation
        for step in range(num_of_steps):
            # aggregation 1
            feature = np.matmul(graph, feature)

            # aggregation 2
            for i in range(len(self.W)):
                feature = self.activf(np.matmul(feature, self.W[i]))

        # readoout
        h_g = np.sum(feature, axis=0)

        return h_g

    def initialize_weight(self, std=0.4) -> None:
        """
        Initialize weight for aggregation

        Parameters
        ----------
        std : float
            standard deviation for normal distribution. The default value is 0.4.

        Returns
        -------
        None

        """

        self.W = []
        for i in range(len(self.layers)-1):
            self.W.append(np.random.normal(scale=std, size=(self.layers[i], self.layers[i+1])))

    def check_shape(self, weight: List[np.ndarray]) -> None:
        """
        Check if the input weight have the correct shape corresponding to self.layers.
        If the shapes are invalid, the ValueError will be raised.

        Parameters
        ----------
        weight : list of array

        Returns
        -------
        None

        """

        invalid_shape = False
        if len(weight) != len(self.layers) - 1:
            invalid_shape = True

        for i in range(len(self.layers)-1):
            w = weight[i]
            if np.shape(w) != (self.layers[i], self.layers[i+1]):
                invalid_shape = True
                break

        if invalid_shape:
            raise ValueError("Invalid size for weight. The size of weight should be the same as self.layers")

    def set_weight(self, weight: List[np.ndarray]) -> None:
        """
        Set weight for aggregation.

        Parameters
        ----------
        weight : list of array
            list of array with size (self.layers[i], self.layers[i+1]), whose length is len(self.layers) - 1.
            Weight for aggregation.

        Returns
        -------
        None

        """

        self.check_shape(weight)
        self.W = np.copy(weight)

    def update_weight(self, diff: List[np.ndarray]) -> None:
        """
        Update the weight by adding given ``diff''. This method is used by optimizers.
        For example, ``diff'' from SGD is the negative gradient multiplied by learning rate.

        Parameters
        ----------
        diff : list of array
            The list of array with size (self.layers[i], self.layers[i+1]), whose length is len(self.layers) - 1.
            This value will be added to the weight.

        Returns
        -------
        None

        """

        self.check_shape(diff)

        for i in range(len(self.W)):
            self.W[i] += diff[i]

    def initialize_features(self, vertices_num: int) -> np.ndarray:
        """
        Initialize feature vector as explained in the problem statement.
        The first element of each feature vector is one,
        and other elements are initialized wish zero.

        Parameters
        ----------
        vertices_num : int
            the number of vertices in the raph

        Returns
        -------
        array
            The feature. Array with size (vertices_num, self.feature_dim)

        """

        feature = np.zeros([vertices_num, self.feature_dim])
        feature[:, 0] = 1.0
        return feature

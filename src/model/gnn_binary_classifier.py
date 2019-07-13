import copy
from typing import List, Tuple

import numpy as np

from model.tools import Params
from model.gnn_graph_feature import GnnGraphFeature
from model.funcs import sigmoid, binary_cross_entropy_with_logit


class GnnBinaryClassifier(object):
    """
    The class for binary classifier using GNN
    Perform binary classification by using features from class GnnGraphFeature.

    """

    def __init__(self, params: Params) -> None:
        """
        Parameters
        ----------
        params : class Params

        """

        self.feature = GnnGraphFeature(params)
        self.feature_dim = params.feature_dim
        self.weight = None  # weight for classifier
        self.bias = None  # bias for classifier

    def __call__(self, graph: np.ndarray) -> float:
        """
        Perform classification for input graph

        Parameters
        ----------
        graph : array
            square and symmetric array which represent a graph as an adjacent matrix

        Returns
        -------
        float
            the probability that the label is 1 (if p > 0.5, the graph is classified as 1, otherwise 0)

        """

        s = self.logit(graph)
        p = sigmoid(s)
        return p

    def prediction(self, graph: np.ndarray) -> int:
        """
        Predict the label for given graph

        Parameters
        ----------
        graph : array
            square and symmetric array which represent a graph as an adjacent matrix

        Returns
        -------
        int
            predicted label

        """

        p = self(graph)
        if p > 0.5:
            return 1
        else:
            return 0

    def logit(self, graph: np.ndarray) -> float:
        """
        Calculate the inner product of the weight and the feature of graph

        Parameters
        ----------
        graph : array
            square and symmetric array which represent a graph as an adjacent matrix

        Returns
        -------
        float
            sigmoid(s) is the probability that the label is 1 (if p > 0.5, the graph is classified as 1, otherwise 0)

        """

        if self.weight is None:
            raise ValueError("Weight is not initialized")
        if self.bias is None:
            raise ValueError("Bias is not initialized")

        h_g = self.feature(graph)
        s = self.weight.dot(h_g) + self.bias
        return s

    def loss(self, graph: np.ndarray, label: int) -> float:
        """
        Calculate cross entropy loss

        Parameters
        ----------
        graph : array
            array with shape (feature_dim, feature_dim) which represent a graph as an adjacent matrix
        label : int
            The correct label

        Returns
        -------
        float
            Cross entropy loss

        """

        s = self.logit(graph)
        xent_loss = binary_cross_entropy_with_logit(s, label)
        return xent_loss

    def initialize_weight(self, std=0.4) -> None:
        """
        Initialize weight and bias for classifier, and weight for graph feature.
        Both of the weights are initialized by normal distribution, and the bias is initialized to 0.

        Parameters
        ----------
        std : float
            standard deviation for normal distribution. The default value is 0.4.

        Returns
        -------
            None

        """

        self.weight = np.random.normal(scale=std, size=self.feature_dim)
        self.bias = 0.0
        self.feature.initialize_weight(std)

    def set_weight(self, weight: np.ndarray, bias: float) -> None:
        """
        Set weight and bias for classifier

        Parameters
        ----------
        weight : array
            array with size (feature_dim)
        bias : float

        Returns
        -------
        None

        """

        if np.shape(weight)[0] != self.feature_dim:
            raise ValueError("The shape of weight should be the same as feature_dim")

        self.weight = np.copy(weight)
        self.bias = np.copy(bias)

    def numerical_derivative_single(self, graph: np.ndarray, label: int, original_loss: float,
                                    derivative_epsilon=0.001) -> float:
        """
        This method will be called in self.numerical_derivative
        Calculate the numerical derivative of one parameter for cross entropy loss.
        It is assumed that weights are already updated (epsilon is added to the corresponding weight)

        Parameters
        ----------
        graph : array
            array with shape (feature_dim, feature_dim) which represent a graph as an adjacent matrix
        label : int
            The correct label
        original_loss : float
            The original loss (before epsilon is added)
        derivative_epsilon : float
            Perturbation for numerical differentiation

        Returns
        -------
        grad : float
            The gradient of the corresponding weight.

        """

        loss = self.loss(graph, label)  # loss with new weight
        grad = (loss - original_loss) / derivative_epsilon  # calculate derivative
        return grad

    def numerical_derivative(self, graph: np.ndarray, label: int, derivative_epsilon=0.001) \
            -> Tuple[np.ndarray, float, List[np.ndarray]]:
        """
        Calculate numerical derivative for cross entropy loss.
        This method calculate derivative for all weights.

        Parameters
        ----------
        graph : array
            array with shape (feature_dim, feature_dim) which represent a graph as an adjacent matrix
        label : int
            The correct label
        derivative_epsilon : float
            Perturbation for numerical differentiation

        Returns
        -------
        grad_a : array
            array with shape (feature_dim), derivative of weights of classifier
        grad_bias : float
            derivative of bias of classifier
        grad_w : array
            derivative of weights of graph feature (GnnGraphFeature)

        """

        # store the current parameters
        original_weight = np.copy(self.weight)
        original_bias = float(np.copy(self.bias))
        original_feature_weight = copy.copy(self.feature.W)
        original_loss = self.loss(graph, label)

        grad_a = np.zeros(np.shape(self.weight))
        grad_w = []
        for i in range(len(self.feature.W)):
            grad_w.append(np.zeros(np.shape(self.feature.W[i])))

        # weight for classifier
        for i in range(self.feature_dim):
            weight = np.copy(original_weight)
            weight[i] += derivative_epsilon  # new weight
            self.set_weight(weight, original_bias)  # set new weight
            grad_a[i] = self.numerical_derivative_single(graph, label, original_loss, derivative_epsilon)
        self.set_weight(original_weight, original_bias)  # restore the weight

        # bias for classifier
        self.set_weight(original_weight, original_bias + derivative_epsilon)  # set new bias
        grad_bias = self.numerical_derivative_single(graph, label, original_loss, derivative_epsilon)
        self.set_weight(original_weight, original_bias)  # restore the bias

        # weight for graph feature
        for n in range(len(grad_w)):
            for i in range(np.shape(grad_w[n])[0]):
                for j in range(np.shape(grad_w[n])[1]):
                    feature_weight = copy.copy(original_feature_weight)
                    feature_weight[n][i, j] += derivative_epsilon  # new weight
                    self.feature.set_weight(feature_weight)  # set new weight
                    grad_w[n][i, j] = self.numerical_derivative_single(graph, label, original_loss, derivative_epsilon)

        # restore the original parameters
        self.set_weight(original_weight, original_bias)
        self.feature.set_weight(original_feature_weight)

        return grad_a, grad_bias, grad_w

    def update(self, graphs: List[np.ndarray], labels: List[int], optimizer, batch_size=1,
               learning_rate=0.0001, derivative_epsilon=0.001) -> None:
        """
        Update parameters (weight and bias for classification, weight for graph feature)

        Parameters
        ----------
        graphs : list of array
            array with shape (feature_dim, feature_dim) which represent a graph as an adjacent matrix
        labels : list of int
            The correct label
        optimizer : class of optimizers
            Optimizers. "sgd" and "momentum_sgd" are implemented.
        batch_size : int
            The size of batch
        learning_rate : float
        derivative_epsilon : float
            Epsilon for numerical derivative

        Returns
        -------
        None

        """

        if len(graphs) != batch_size:
            raise ValueError("The graph should be a list of array, whose length is equal to batch size")
        if len(labels) != batch_size:
            raise ValueError("The label should be a list of int, whose length is equal to batch size")

        # compute batch gradient
        grad_a_sum = np.zeros(np.shape(self.weight))
        grad_bias_sum = 0.0
        grad_w_sum = np.zeros(np.shape(self.feature.W))
        for graph, label in zip(graphs, labels):
            grad_a_, grad_bias_, grad_w_ = self.numerical_derivative(graph, label, derivative_epsilon)
            grad_a_sum += grad_a_
            grad_bias_sum += grad_bias_
            grad_w_sum += grad_w_

        grad_a = grad_a_sum / batch_size
        grad_bias = grad_bias_sum / batch_size
        grad_w = grad_w_sum / batch_size

        # the update values are calculated by optimizer
        optimizer.set_learning_rate(learning_rate)
        diff_a, diff_bias, diff_w = optimizer(grad_a, grad_bias, grad_w)

        self.weight += diff_a
        self.bias += diff_bias
        self.feature.update_weight(diff_w)

from typing import Tuple

import numpy as np


class SGD(object):
    """
    A class for SGD optimizer.
    This class is inherited by other optimizers classes.
    """

    def __init__(self, learning_rate: float) -> None:
        """
        Parameters
        ----------
        learning_rate : float

        """

        self.learning_rate = learning_rate

    def __call__(self, grad_a: np.ndarray, grad_bias: float, grad_w: np.ndarray) \
            -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Parameters
        ----------
        grad_a : array
            Array with size (feature_dim). The gradient for the weight of the classifier.
        grad_bias : float
            The gradient for the bias of the classifier
        grad_w : array
            Array with size (feature_dim, feature_dim). The gradient for the weight for graph feature aggregation.

        Returns
        -------
        diff_a : array
            Array with size (feature_dim). The update value the weight of the classifier.
        diff_bias : float
            The update value the bias of the classifier.
        diff_w : array
            Array with size (feature_dim, feature_dim). The update value the weight of the graph feature aggregation.

        """

        return -self.learning_rate * grad_a, -self.learning_rate * grad_bias, -self.learning_rate * grad_w

    def set_learning_rate(self, learning_rate: float) -> None:
        """
        Update learning rate for SGD. This method my be used for learning rate decay.

        Parameters
        ----------
        learning_rate : float

        Returns
        -------
        None

        """

        self.learning_rate = learning_rate


class MomentumSGD(SGD):
    """
    A class for momentum SGD.
    """

    def __init__(self, learning_rate: float, moment: float, feature_dim: int) -> None:
        """
        Parameters
        ----------
        learning_rate : float
        moment : float
            The parameter "mu" of momentum sgd
        feature_dim : int
            The dimension of the feature of each vertex in the graph

        """

        super().__init__(learning_rate)
        self.moment = moment  # parameter mu

        # previous update values will be stored
        self.moment_a = np.zeros([feature_dim])
        self.moment_bias = 0
        self.moment_w = np.zeros([feature_dim, feature_dim])

    def __call__(self, grad_a: np.ndarray, grad_bias: float, grad_w: np.ndarray) \
            -> Tuple[np.ndarray, float, np.ndarray]:
        """

        Parameters
        ----------
        grad_a : array
            Array with size (feature_dim). The gradient for the weight of the classifier.
        grad_bias : float
            The gradient for the bias of the classifier
        grad_w : array
            Array with size (feature_dim, feature_dim). The gradient for the weight for graph feature aggregation.

        Returns
        -------
        diff_a : array
            Array with size (feature_dim). The update value the weight of the classifier.
        diff_bias : float
            The update value the bias of the classifier.
        diff_w : array
            Array with size (feature_dim, feature_dim). The update value the weight of the graph feature aggregation.

        """

        sgd_a, sgd_bias, sgd_w = super().__call__(grad_a, grad_bias, grad_w)

        diff_a = sgd_a + self.moment * self.moment_a
        diff_bias = sgd_bias + self.moment * self.moment_bias
        diff_w = sgd_w + self.moment * self.moment_w

        # update moment
        self.moment_a = np.copy(diff_a)
        self.moment_bias = np.copy(diff_bias)
        self.moment_w = np.copy(diff_w)

        return diff_a, diff_bias, diff_w


class Adam(SGD):
    """
    A class for optimizer Adam.
        Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." ICLR 2014.

    """

    def __init__(self, learning_rate: float, feature_dim: int,
                 beta1=0.9, beta2=0.999, epsilon=1e-8) -> None:
        """
        Initialize class Adam

        Parameters
        ----------
        learning_rate : float
        feature_dim : int
            The dimension of the feature of each vertex in the graph
        beta1 : float
            The parameter for 1st moment update
        beta2 : float
            The parameter for 2nd moment update
        epsilon : float
            The parameter for Adam

        """

        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # 1st moment
        self.moment_a = np.zeros([feature_dim])
        self.moment_bias = 0
        self.moment_w = np.zeros([feature_dim, feature_dim])

        # 2nd moment
        self.v_a = np.zeros([feature_dim])
        self.v_bias = 0
        self.v_w = np.zeros([feature_dim, feature_dim])

        self.t = 0  # time step

    def __call__(self,  grad_a: np.ndarray, grad_bias: float, grad_w: np.ndarray) \
            -> Tuple[np.ndarray, float, np.ndarray]:
        """

        Parameters
        ----------
        grad_a : array
            Array with size (feature_dim). The gradient for the weight of the classifier.
        grad_bias : float
            The gradient for the bias of the classifier
        grad_w : array
            Array with size (feature_dim, feature_dim). The gradient for the weight for graph feature aggregation.

        Returns
        -------
        diff_a : array
            Array with size (feature_dim). The update value the weight of the classifier.
        diff_bias : float
            The update value the bias of the classifier.
        diff_w : array
            Array with size (feature_dim, feature_dim). The update value the weight of the graph feature aggregation.

        """

        # increment time step
        self.t += 1

        # update moments
        self.moment_a, self.v_a = self.update_moments(self.moment_a, self.v_a, grad_a)
        self.moment_bias, self.v_bias = self.update_moments(self.moment_bias, self.v_bias, np.array(grad_bias))
        self.moment_w, self.v_w = self.update_moments(self.moment_w, self.v_w, grad_w)

        # bias corrected moment
        mhat_a = self.moment_a / (1.0-np.power(self.beta1, self.t))
        mhat_bias = self.moment_bias / (1.0-np.power(self.beta1, self.t))
        mhat_w = self.moment_w / (1.0-np.power(self.beta1, self.t))
        vhat_a = self.v_a / (1.0-np.power(self.beta2, self.t))
        vhat_bias = self.v_bias / (1.0-np.power(self.beta2, self.t))
        vhat_w = self.v_w / (1.0-np.power(self.beta2, self.t))

        # update value
        diff_a = self.update_value(mhat_a, vhat_a)
        diff_bias = self.update_value(mhat_bias, vhat_bias)
        diff_w = self.update_value(mhat_w, vhat_w)

        return diff_a, diff_bias, diff_w

    def update_moments(self, past_moment1: np.ndarray, past_moment2: np.ndarray, grad: np.ndarray)\
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate updated value of 1st and 2nd moments for Adam

        Parameters
        ----------
        past_moment1 : array
            Current value of 1st moment (m_t in the original paper)
        past_moment2 : array
            Current value of 2nd moment (v_t in the original paper)
        grad : array
            The computed gradient for corresponding parameters

        Returns
        -------
        moment1 : array
            updated value of 1st moment
        moment2 : array
            updated value of 2nd moment

        """

        moment1 = self.beta1 * past_moment1 + (1.0-self.beta1) * grad
        moment2 = self.beta2 * past_moment2 + (1.0-self.beta2) * np.power(grad, 2)
        return moment1, moment2

    def update_value(self, moment1: np.ndarray, moment2: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        moment1 : array
            1st moment (hat{m_t} in the original paper)
        moment2 : array
            2nd moment (hat{v_t} in the original paper)

        Returns
        -------
        array
            the value which will be added to the current parameters

        """

        return - self.learning_rate * moment1 / (np.sqrt(moment2) + self.epsilon)

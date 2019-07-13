import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np

from model.tools import Params
from model.gnn_binary_classifier import GnnBinaryClassifier
from model.optimizer import SGD

if __name__ == "__main__":
    np.random.seed(19951202)
    params = Params()
    optimizer = SGD(params.learning_rate)

    edges = [[0, 1], [0, 3], [3, 4], [3, 5], [3, 8], [5, 8], [6, 9], [7, 8], [7, 9]]
    graph = np.zeros([10, 10])
    for e in edges:
        v1, v2 = e
        graph[v1, v2] = 1.0
        graph[v2, v1] = 1.0
    label = 1

    classifier = GnnBinaryClassifier(params)
    classifier.initialize_weight(std=0.4)
    pred = classifier(graph)
    loss = classifier.loss(graph, label)
    grad_a, grad_b, grad_w = classifier.numerical_derivative(graph, label)

    for itr in range(5000):
        classifier.update([graph], [label], optimizer)
        if itr % 100 == 99:
            pred = classifier(graph)
            loss = classifier.loss(graph, label)
            print("itr: %3d\tloss: %f\tprediction: %f" % ((itr+1), loss, pred))

    print("Testing passed")

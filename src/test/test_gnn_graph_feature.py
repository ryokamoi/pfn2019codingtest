import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np

from model.tools import Params
from model.gnn_graph_feature import GnnGraphFeature


if __name__ == "__main__":
    params = Params()
    params.feature_dim = 2

    graph = np.array([[0, 0, 1, 1],
                      [0, 0, 1, 0],
                      [1, 1, 0, 1],
                      [1, 0, 1, 0]])

    weight = [np.array([[1, 2],
                       [3, 4]])]

    gnn_graph = GnnGraphFeature(params)
    gnn_graph.set_weight(weight)
    output = gnn_graph(graph, num_of_steps=2)
    assert (output == np.array([126, 180])).all()

    print("Testing passed")

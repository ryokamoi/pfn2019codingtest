import argparse
import pickle as pkl

import numpy as np

from model.gnn_binary_classifier import GnnBinaryClassifier
from model.tools import Params
from model.dataset_utils import read_test_dataset, add_super_node


def prediction(configfile, test_data_directory, output_file):
    # read parameters
    params = Params()
    params.read_config(configfile)

    # read dataset
    test_graph = read_test_dataset(test_data_directory)

    # add super node
    if params.super_node:
        test_graph = add_super_node(test_graph)

    # load classifier
    model_file = "/".join(configfile.split("/")[:-1] + ["model.pkl"])
    with open(model_file, "rb") as f:
        classifier = pkl.load(f)

    # test
    predictions = []
    for graph in test_graph:
        predictions.append(classifier.prediction(graph))

    # output results
    with open(output_file, "w") as f:
        for pred in predictions:
            print(pred, file=f)


if __name__ == "__main__":
    np.random.seed(19951202)

    parser = argparse.ArgumentParser(description='Perform prediction for test dataset.')
    parser.add_argument('configfile', type=str,
                        help='A path for configuration file')
    parser.add_argument('test_data_directory', type=str,
                        help='A path for a directory of test data')
    parser.add_argument('output_file', type=str,
                        help='A path for output file')
    args = parser.parse_args()

    prediction(args.configfile, args.test_data_directory, args.output_file)

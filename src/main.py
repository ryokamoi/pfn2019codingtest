import os
import argparse
import shutil
import pickle as pkl
from copy import copy
from datetime import datetime

import numpy as np

from model.gnn_binary_classifier import GnnBinaryClassifier
from model.optimizer import SGD, MomentumSGD, Adam
from model.funcs import evaluation
from model.tools import Params, print_write
from model.dataset_utils import read_train_dataset, add_super_node, shuffle_and_split_datset


def main(configfile_name):
    # read parameters
    params = Params()
    params.read_config(configfile_name)

    # output directory
    now = datetime.now().strftime("%y%m%d-%H%M%S")
    logdir = "output/" + now + "-%s" % params.optimizer
    os.makedirs(logdir)
    shutil.copy(configfile_name, logdir + "/config.txt")
    output_file = logdir + "/result.txt"

    # read dataset
    train_graph, train_label, valid_graph, valid_label = read_train_dataset(params.train_data_directory)

    # add super node
    if params.super_node:
        train_graph = add_super_node(train_graph)
        valid_graph = add_super_node(valid_graph)

    # initialize classifier
    classifier = GnnBinaryClassifier(params)
    classifier.initialize_weight(std=params.initialization_std)

    # set optimizer
    if params.optimizer == "sgd":
        optimizer = SGD(params.learning_rate)
    elif params.optimizer == "momentum_sgd":
        optimizer = MomentumSGD(params.learning_rate, params.moment, params.feature_dim)
    elif params.optimizer == "adam":
        optimizer = Adam(params.learning_rate, params.feature_dim)
    else:
        raise ValueError("The optimizer '{}' is not implemented" % params.optimizer)

    # training
    best_classifier = None
    best_results = [0, 0, 1e+9, 0, 1e+9]  # epoch, validation accuracy, loss, training accuracy, loss
    for epoch in range(1, params.num_of_epochs+1):
        epoch_dataset_graphs, epoch_dataset_labels = \
            shuffle_and_split_datset(train_graph, train_label, params.batch_size)

        # update parameters
        for graphs, labels in zip(epoch_dataset_graphs, epoch_dataset_labels):
            classifier.update(graphs, labels, optimizer=optimizer,
                              batch_size=params.batch_size, learning_rate=params.learning_rate,
                              derivative_epsilon=params.derivative_epsilon)

        # evaluate training and validation loss, accuracy of this epoch
        train_accuracy, train_loss = evaluation(classifier, train_graph, train_label)
        valid_accuracy, valid_loss = evaluation(classifier, valid_graph, valid_label)
        print_write("epoch:%3d\tvalid accuracy:%.5f\tvalid loss:%.5f\ttrain accuracy:%.5f\ttrain loss:%.5f\t" %
                    (epoch, valid_accuracy, valid_loss, train_accuracy, train_loss), output_file)

        # Store the best model (in terms of the validation loss)
        if best_results[2] > valid_loss:
            best_results = [epoch, valid_accuracy, valid_loss, train_accuracy, train_loss]
            best_classifier = copy(classifier)

    # Save the best model
    print_write("best result:\tepoch:%3d\tvalid accuracy:%.5f\t"
                "valid loss:%.5f\ttrain accuracy:%.5f\ttrain loss:%.5f\t" %
                (best_results[0], best_results[1], best_results[2], best_results[3], best_results[4]), output_file)
    with open(logdir + "/model.pkl", "wb") as f:
        pkl.dump(best_classifier, f)


if __name__ == "__main__":
    np.random.seed(19951202)

    parser = argparse.ArgumentParser(description='Train binary classification with Graphical Neural Networks.')
    parser.add_argument('configfile', type=str,
                        help='A path for configuration file')
    args = parser.parse_args()

    main(args.configfile)

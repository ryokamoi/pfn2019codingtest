from typing import List, Tuple
import random

import numpy as np

random.seed(19951202)


def read_graph_file(filename: str) -> np.ndarray:
    """
    read one graph from a file

    Parameters
    ----------
    filename : str

    Returns
    -------
    array
        graph

    """

    with open(filename, "r") as f:
        feature_size = int(f.readline()[:-1])

        graph = np.zeros([feature_size, feature_size])
        for i in range(feature_size):
            line = f.readline()[:-1]
            graph[i] = np.array(list(map(float, line.split())))
    return graph


def read_label_file(filename: str) -> int:
    """
    read one label from a file

    Parameters
    ----------
    filename : str

    Returns
    -------
    int
        label

    """

    with open(filename, "r") as f:
        label = int(f.readline()[:-1])
    return label


def read_graph_files(filenames: List[str]) -> List[np.ndarray]:
    """
    read multiple graphs from files

    Parameters
    ----------
    filenames : list of str
        list of filenames of graphs

    Returns
    -------
    list of array
        list of graphs

    """

    graphs = []
    for filename in filenames:
        graph = read_graph_file(filename)
        graphs.append(graph)
    return graphs


def read_label_files(filenames: List[str]) -> List[int]:
    """
    read multiple labels from files

    Parameters
    ----------
    filenames : list of str
        list of filenames of labels

    Returns
    -------
    list of int
        list of labels

    """

    labels = []
    for filename in filenames:
        label = read_label_file(filename)
        labels.append(label)
    return labels


def read_train_dataset(data_directory: str, num_of_dataset=2000) \
        -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
    """
    Read dataset from given directory.
    The names of each file should be given as "%d_graph.txt" and "%d_label.txt",
    where the numbers are zero to num_of_dataset-1.
    75% of the dataset are used for training, and 25% of the datsaet are usd for validation

    Parameters
    ----------
    data_directory : str
        the directory name for training dataset
    num_of_dataset : int
        the number fo dataset

    Returns
    -------
    train_graph : list of array
        graphs for training
    train_label : list of int
        labels for training
    valid_graph : list of array
        graphs for validation
    valid_label : list of int
        labels for validation
    """

    num_of_training_data = int(num_of_dataset * 0.75)
    train_graph_filenames = [data_directory + "/%d_graph.txt" % i for i in range(num_of_training_data)]
    valid_graph_filenames = [data_directory + "/%d_graph.txt" % i for i in range(num_of_training_data, num_of_dataset)]
    train_label_filenames = [data_directory + "/%d_label.txt" % i for i in range(num_of_training_data)]
    valid_label_filenames = [data_directory + "/%d_label.txt" % i for i in range(num_of_training_data, num_of_dataset)]

    train_graph = read_graph_files(train_graph_filenames)
    train_label = read_label_files(train_label_filenames)
    valid_graph = read_graph_files(valid_graph_filenames)
    valid_label = read_label_files(valid_label_filenames)

    return train_graph, train_label, valid_graph, valid_label


def read_test_dataset(data_directory: str, num_of_dataset=500) -> List[np.ndarray]:
    """
    Read dataset for test.

    Parameters
    ----------
    data_directory : str
        the directory name for test dataset
    num_of_dataset : int
        the number fo dataset

    Returns
    -------
    graph : list of array
        graphs for test
    """

    filenames = [data_directory + "/%d_graph.txt" % i for i in range(num_of_dataset)]
    graph = read_graph_files(filenames)
    return graph


def add_super_node(list_of_graphs: List[np.ndarray]) -> List[np.ndarray]:
    """
    Add super node to the list of graphs (dataset).

    Parameters
    ----------
    list_of_graphs : list of array
        list of graphs

    Returns
    -------
    output : list of array
        list of new graphs

    """

    output = []
    for graph in list_of_graphs:
        new_graph = np.copy(graph)
        new_graph = np.vstack([new_graph, np.ones(np.shape(new_graph)[1])])
        new_graph = np.hstack([new_graph, np.ones([np.shape(new_graph)[0], 1])])
        new_graph[-1, -1] = 0
        output.append(new_graph)
    return output


def shuffle_and_split_datset(graphs: List[np.ndarray], labels: List[int], batch_size: int) \
        -> Tuple[List[List[np.ndarray]], List[List[int]]]:
    """
    Shuffle and split graphs and labels for batch training.

    Parameters
    ----------
    graphs : list of array
    labels : list of int
    batch_size : int

    Returns
    -------
    graphs : list of list of array
    labels : list of list of int

    """

    dataset_size = len(graphs)
    if dataset_size % batch_size != 0:
        raise ValueError("batch size should be a divisor of the number of dataset")

    idxes = [i for i in range(dataset_size)]
    random.shuffle(idxes)
    idxes = np.reshape(idxes, [-1, batch_size]).tolist()
    graphs = [[graphs[i] for i in idx] for idx in idxes]
    labels = [[labels[i] for i in idx] for idx in idxes]
    return graphs, labels

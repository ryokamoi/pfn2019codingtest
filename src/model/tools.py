from typing import Union


def convert_type(s: str) -> Union[int, float, bool, str]:
    """
    Convert the type from string to int, float or bool.
    This function will be used in class Params.

    Parameters
    ----------
    s : str

    Returns
    -------
    int or float or bool or str

    """
    if s in ["True", "False"]:
        if s == "True":
            return True
        else:
            return False
    try:
        float(s)
        if "." in s:
            return float(s)
        else:
            return int(s)
    except ValueError:
        return s


class Params(object):
    """
    A class of parameters. All parameters specified in configuration file will be stored in this class.
    """

    def __init__(self) -> None:
        """
        Initialize parameters by initial values. The values will be updated by "read_config".
        """

        self.optimizer = "momentum_sgd"
        self.aggregation_num = 2
        self.feature_dim = 8
        self.layers = None
        self.learning_rate = 0.001
        self.moment = 0.9
        self.initialization_std = 0.4
        self.derivative_epsilon = 0.001
        self.num_of_epochs = 50
        self.batch_size = 5
        self.activation_for_graph_feature = "relu"
        self.train_data_directory = "datasets/train"
        self.super_node = False

    def read_config(self, configfile: str) -> None:
        """
        Read configuration file and update parameters

        Parameters
        ----------
        configfile : text
            The path to configuration file

        Returns
        -------
        None

        """

        with open(configfile, "r") as f:
            for l in f.readlines():
                line = l[:-1]
                varname, value = line.split("=")
                if value[0] == "[" and value[-1] == "]":
                    cmd = "self.%s = list(map(convert_type, '%s'[1:-1].split(',')))" % (varname, value)
                else:
                    cmd = "self.%s = convert_type('%s')" % (varname, value)
                exec(cmd)


def print_write(text: str, filename: str) -> None:
    """
    Print text and write the same text to a specified file

    Parameters
    ----------
    text : str
    filename : str

    Returns
    -------
    None

    """

    with open(filename, "a") as f:
        print(text)
        print(text, file=f)

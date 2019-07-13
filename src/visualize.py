import argparse

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def visualization(logfile):
    logdir = "/".join(logfile.split("/")[:-1])

    # read results
    epoch = []
    valid_acc = []
    valid_loss = []
    train_acc = []
    train_loss = []
    with open(logfile, "r") as f:
        for l in f.readlines():
            line = l.split("\t")
            if line[0] == "best result:":
                break
            epoch.append(int(line[0].split(":")[1]))
            valid_acc.append(float(line[1].split(":")[1]))
            valid_loss.append(float(line[2].split(":")[1]))
            train_acc.append(float(line[3].split(":")[1]))
            train_loss.append(float(line[4].split(":")[1]))

    # visualize loss
    loss = plt.figure()
    plt.plot(epoch, train_loss, label="training loss")
    plt.plot(epoch, valid_loss, label="validation loss")
    plt.ylim([0.60, 0.75])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(logdir + "/loss.png")

    # visualize accuracy
    acc = plt.figure()
    plt.plot(epoch, train_acc, label="training accuracy")
    plt.plot(epoch, valid_acc, label="validation accuracy")
    plt.ylim([0.45, 0.7])
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(logdir + "/accuracy.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw plots for loss and accuracy.')
    parser.add_argument('logfile', type=str,
                        help='A path for log file (result.txt)')
    args = parser.parse_args()

    visualization(args.logfile)

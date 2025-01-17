import numpy as np
import os 

from sklearn.datasets import make_swiss_roll, load_digits 
import graphlearning as gl

DATALOC = "./data"
if __name__ == "__main__":
    if not os.path.exists(DATALOC):
        os.makedirs(DATALOC)

    if not os.path.exists(os.path.join(DATALOC, "swissroll.npz")):
        X, y = make_swiss_roll(n_samples=500)
        print("SwissRoll", X.shape, y.shape)
        np.savez(os.path.join(DATALOC, "swissroll.npz"), X=X, y=y)


    if not os.path.exists(os.path.join(DATALOC, "digits.npz")):
        X, y = load_digits(return_X_y=True)
        print("Digits", X.shape, y.shape)
        np.savez(os.path.join(DATALOC, "digits.npz"), X=X, y=y)


    if not os.path.exists(os.path.join(DATALOC, "mnist.npz")):
        X, y = gl.datasets.load("mnist", metric="raw")
        print("MNIST", X.shape, y.shape)
        np.savez(os.path.join(DATALOC, "mnist.npz"), X=X, y=y)
        os.remove(os.path.join(DATALOC, "mnist_labels.npz"))
        os.remove(os.path.join(DATALOC, "mnist_raw.npz"))

    if not os.path.exists(os.path.join(DATALOC, "fashionmnist.npz")):
        X, y = gl.datasets.load("fashionmnist", metric="raw")
        print("FashionMNIST", X.shape, y.shape)
        np.savez(os.path.join(DATALOC, "fashionmnist.npz"), X=X, y=y)
        os.remove(os.path.join(DATALOC, "fashionmnist_labels.npz"))
        os.remove(os.path.join(DATALOC, "fashionmnist_raw.npz"))

    if not os.path.exists(os.path.join(DATALOC, "cifar10.npz")):
        X, y = gl.datasets.load("cifar10", metric="raw")
        print("CIFAR10", X.shape, y.shape)
        np.savez(os.path.join(DATALOC, "cifar10.npz"), X=X, y=y)
        os.remove(os.path.join(DATALOC, "cifar10_labels.npz"))
        os.remove(os.path.join(DATALOC, "cifar10_raw.npz"))
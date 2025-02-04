import numpy as np
import os 

from sklearn.datasets import make_swiss_roll, load_digits 
import cv2
import fiftyone.zoo as foz
import fiftyone as fo 

fo.config.dataset_zoo_dir = "./fiftyone_data/"

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

    if not os.path.exists(os.path.join(DATALOC, "mnist_test.npz")):
        dataset = foz.load_zoo_dataset("mnist")
        data_split = dataset.match_tags("test")
        X = np.array([
            cv2.imread(f, cv2.IMREAD_UNCHANGED).ravel() for f in data_split.values("filepath")
        ])
        y = np.array([int(val['label'][0]) for val in data_split.values("ground_truth")])
        print("MNIST (TEST SPLIT)", X.shape, y.shape)
        np.savez(os.path.join(DATALOC, "mnist_test.npz"), X=X, y=y)
    
    if not os.path.exists(os.path.join(DATALOC, "fashion-mnist_test.npz")):
        dataset = foz.load_zoo_dataset("fashion-mnist")
        data_split = dataset.match_tags("test")
        X = np.array([
            cv2.imread(f, cv2.IMREAD_UNCHANGED).ravel() for f in data_split.values("filepath")
        ])
        labels = np.array([val['label'] for val in data_split.values("ground_truth")])
        classes = np.unique(labels)
        label2class = {classes[i] : i for i in range(classes.size)}
        y = np.array([label2class[lab] for lab in labels])
        print("fashion-mnist (TEST SPLIT)", X.shape, y.shape)
        np.savez(os.path.join(DATALOC, "fashion-mnist_test.npz"), X=X, y=y)

    if not os.path.exists(os.path.join(DATALOC, "cifar10.npz")):
        dataset = foz.load_zoo_dataset("cifar10")
        data_split = dataset.match_tags("test")
        X = np.array([
            cv2.imread(f, cv2.IMREAD_UNCHANGED).ravel() for f in data_split.values("filepath")
        ])
        labels = np.array([val['label'] for val in data_split.values("ground_truth")])
        classes = np.unique(labels)
        label2class = {classes[i] : i for i in range(classes.size)}
        y = np.array([label2class[lab] for lab in labels])
        print("CIFAR10 (TEST SPLIT)", X.shape, y.shape)
        np.savez(os.path.join(DATALOC, "cifar10_test.npz"), X=X, y=y)

    if not os.path.exists(os.path.join(DATALOC, "mnist.npz")):
        dataset = foz.load_zoo_dataset("mnist")
        data_split = dataset.view()
        X = np.array([
            cv2.imread(f, cv2.IMREAD_UNCHANGED).ravel() for f in data_split.values("filepath")
        ])
        y = np.array([int(val['label'][0]) for val in data_split.values("ground_truth")])
        print("MNIST", X.shape, y.shape)
        np.savez(os.path.join(DATALOC, "mnist.npz"), X=X, y=y)

    

    # if not os.path.exists(os.path.join(DATALOC, "fashion-mnist.npz")):
    #     dataset = foz.load_zoo_dataset("fashion-mnist")
    #     data_split = dataset.view()
    #     X = np.array([
    #         cv2.imread(f, cv2.IMREAD_UNCHANGED).ravel() for f in data_split.values("filepath")
    #     ])
    #     y = np.array([int(val['label'][0]) for val in data_split.values("ground_truth")])
    #     print("FASIONMNIST", X.shape, y.shape)
    #     np.savez(os.path.join(DATALOC, "fashion-mnist.npz"), X=X, y=y)

    # if not os.path.exists(os.path.join(DATALOC, "cifar10.npz")):
    #     dataset = foz.load_zoo_dataset("cifar10")
    #     data_split = dataset.view()
    #     X = np.array([
    #         cv2.imread(f, cv2.IMREAD_UNCHANGED).ravel() for f in data_split.values("filepath")
    #     ])
    #     y = np.array([int(val['label'][0]) for val in data_split.values("ground_truth")])
    #     print("CIFAR10", X.shape, y.shape)
    #     np.savez(os.path.join(DATALOC, "cifar10.npz"), X=X, y=y)
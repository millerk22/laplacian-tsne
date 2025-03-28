import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from argparse import ArgumentParser
import pandas as pd

import os
import shutil
from tqdm import tqdm 
from collections import Counter
 

###############################################
###############################################
##### code copied from https://github.com/YingfanWang/PaCMAP/blob/master/evaluation/evaluation.py
##### (from the Rudin 2020 tSNE paper)
##############################################

def knn_clf(nbr_vec, y):
    '''
    Helper function to generate knn classification result.
    '''
    y_vec = y[nbr_vec]
    c = Counter(y_vec)
    return c.most_common(1)[0][0]
    
def knn_eval(X, y, n_neighbors=1):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An accuracy is calculated by an k-nearest neighbor classifier.
    Input:
        X: A numpy array with the shape [N, k]. The lower dimension embedding
           of some dataset. Expected to have some clusters.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset.
        kwargs: Any keyword argument that is send into the knn clf.
    Output:
        acc: The avg accuracy generated by the clf, using leave one out cross val.
    '''
    sum_acc = 0
    max_acc = X.shape[0]
    # Train once, reuse multiple times
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    indices = indices [:, 1:]
    distances = distances[:, 1:]
    for i in range(X.shape[0]):
        result = knn_clf(indices[i], y)
        if result == y[i]:
            sum_acc += 1
    avg_acc = sum_acc / max_acc
    return avg_acc

def knn_eval_series(X, y, n_neighbors_list=[1, 3, 5, 10, 15, 20, 25, 30]):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An accuracy is calculated by an k-nearest neighbor classifier.
    A series of accuracy will be calculated for the given n_neighbors.
    Input:
        X: A numpy array with the shape [N, k]. The lower dimension embedding
           of some dataset. Expected to have some clusters.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset.
        n_neighbors_list: A list of int.
        kwargs: Any keyword argument that is send into the knn clf.
    Output:
        accs: The avg accuracy generated by the clf, using leave one out cross val.
    '''
    avg_accs = []
    for n_neighbors in n_neighbors_list:
        avg_acc = knn_eval(X, y, n_neighbors)
        avg_accs.append(avg_acc)
    return avg_accs

def faster_knn_eval_series(X, y, n_neighbors_list=[1, 3, 5, 10, 15, 20, 25, 30]):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An accuracy is calculated by an k-nearest neighbor classifier.
    A series of accuracy will be calculated for the given n_neighbors.
    Input:
        X: A numpy array with the shape [N, k]. The lower dimension embedding
           of some dataset. Expected to have some clusters.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset.
        n_neighbors_list: A list of int.
        kwargs: Any keyword argument that is send into the knn clf.
    Output:
        accs: The avg accuracy generated by the clf, using leave one out cross val.
    '''
    avg_accs = []
    max_acc = X.shape[0]
    # Train once, reuse multiple times
    nbrs = NearestNeighbors(n_neighbors=n_neighbors_list[-1]+1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    indices = indices [:, 1:]
    distances = distances[:, 1:]
    for n_neighbors in n_neighbors_list:
        sum_acc = 0
        for i in range(X.shape[0]):
            indices_temp = indices[:, :n_neighbors]
            result = knn_clf(indices_temp[i], y)
            if result == y[i]:
                sum_acc += 1
        avg_acc = sum_acc / max_acc
        avg_accs.append(avg_acc)
    return avg_accs


def evaluate_output(X, X_new, y, baseline=False):
    if baseline:
        baseline_knn_accs = faster_knn_eval_series(X, y)
    else:
        baseline = None
    knn_accs = faster_knn_eval_series(X_new, y)
    
    return knn_accs, baseline_knn_accs
###################
#### As add different evaluations, add them to harcoded list of evaluations used in this evaluate_output function 
#### MAKE SURE THE ORDERING IS CORRECT FOR WHAT IS RETURNED IN evaluate_output function
EVALUATIONS = ["knn class acc", "baseline knn class acc"]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--loc", type=str, default="./results/")
    parser.add_argument("--datasets", nargs="+", default=[])
    parser.add_argument("--recompute", type=int, default=0)
    args = parser.parse_args()

    # load the results for the datasets indicated. if list is empty, run for all datasets found in args.loc 
    assert os.path.exists(args.loc)

    datasets_given = args.datasets

    # check that these datasets have original datasets under the 'data' directory
    for i, d in enumerate(datasets_given):
        if not os.path.exists(f"./data/{d}.npz"):
            print(f"WARNING: Could not find orig. data for {d} under the data directory (./data/)...")
            datasets_given[i] = "notpresent"

    datasets_loc = [f.path for f in os.scandir(args.loc) if f.is_dir()] 
    if len(datasets_given) == 0:
        datasets_to_do = datasets_loc
    else:
        datasets_to_do = []
        for d in datasets_given:
            if os.path.join(args.loc, d) in datasets_loc:
               datasets_to_do.append(os.path.join(args.loc, d)) 
    print(f"Found the following datasets' results to evaluate in {args.loc}:")
    for dirname in datasets_to_do:
        print(f"\t{dirname}")
    print("")

    
    for idx, dirname in enumerate(datasets_to_do):
        dataset_name = dirname.split("/")[-1]

        # load the original dataset, with associated class labels, y
        data = np.load(os.path.join("./data", f"{dataset_name}.npz"), allow_pickle=True)
        X, y = data['X'], data['y']
        if y.dtype != int:
            print(f"Labels (y) for dataset = {dataset_name} are not integers, skipping...")
            continue

        # evaluate each of the learned embeddings that don't already have results in this directory
        list_of_X_ = [fname for fname in os.listdir(dirname) if fname.split(".")[-1] == "npy"]
        savename = os.path.join(dirname, "eval.csv")

        if os.path.exists(savename) and not args.recompute:
            # if have previous results, load that dataframe in
            res_df = pd.read_csv(savename)
            already_evaluated = res_df["name"].values
        else:
            # no previously saved results, instantiate new dataframe
            res_df = pd.DataFrame(columns=["name"]+EVALUATIONS)
            already_evaluated = np.array([])
            
        if not args.recompute:
            list_of_X_ = np.setdiff1d(list_of_X_, already_evaluated)
        
        for i, fname in tqdm(enumerate(list_of_X_), total=len(list_of_X_), desc=f"Evaluating embeddings for {dataset_name}, ({idx+1}/{len(datasets_to_do)})"):
            X_ = np.load(os.path.join(dirname, fname))
            knn_acc, baseline_knn_acc = evaluate_output(X, X_, y, baseline=True)
            res_df.loc[len(res_df)+1] = [fname, knn_acc, baseline_knn_acc]
            res_df.to_csv(savename, index=None)
            

            

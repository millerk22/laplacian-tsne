import os 
import yaml 
from itertools import product
from argparse import ArgumentParser 
from tqdm import tqdm
from time import time
import pandas as pd

from lap_tsne import *


def run_experiment(X, LapTSNE, repulsion_rel_weight=1.0, repulsion_kernel='hat', num_landmarks=100, hat_bandwidth=None):
    # manually set these hyperparameters for the already prepped object 
    LapTSNE.gamma = repulsion_rel_weight
    LapTSNE.repulsion_kernel = repulsion_kernel 
    LapTSNE.num_landmarks = num_landmarks 
    LapTSNE.hat_bandwidth = hat_bandwidth

    tic = time()
    X_embedded = LapTSNE.fit_transform(X)
    toc = time()
    run_time = toc - tic

    return X_embedded, run_time



if __name__ == "__main__":
    parser = ArgumentParser(description="Main testing script for running LapTSNE.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resultsdir", type=str, default="./results/")
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.resultsdir):
        print(f"Creating results directory, {args.resultsdir}")
        os.makedirs(args.resultsdir)
    
    # load in configuration file
    assert os.path.exists(args.config)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # fixed values for each test
    m = config["m"]                     # target dimension
    perplexity = config["perplexity"]   
    dataset = config["dataset"]
    learning_rate = config["learning_rate"]
    approx_nn = config["approx_nn"]     # whether or not to do approximate nearest neighbor search in graph construction
    assert type(m) == int 
    assert type(perplexity) != list 
    assert type(approx_nn) != list 

    knn_graph_vals = config["knn_graph"]
    k_eigen_vals = config["k_eigen"]
    assert type(knn_graph_vals) == list 
    assert type(k_eigen_vals) == list 
    graph_setting_list = list(product(knn_graph_vals, k_eigen_vals))

    rep_kernel_vals = config["repulsion_kernel"]
    rep_rel_weights = config["repulsion_rel_weights"]
    num_lm_vals = config["num_landmarks"]
    hat_bandwidth_vals = config["hat_bandwidth"]
    assert type(rep_kernel_vals) == list 
    assert type(rep_rel_weights) == list 
    assert type(num_lm_vals) == list 
    assert type(hat_bandwidth_vals) == list
    opt_hyperparam_list = []
    for tup in list(product(rep_rel_weights, rep_kernel_vals, num_lm_vals)):
        if tup[1] == "hat":
            opt_hyperparam_list.extend([(tup[0], tup[1], tup[2], hat_bw) for hat_bw in hat_bandwidth_vals])
        else:
            opt_hyperparam_list.append((tup[0], tup[1], tup[2], None))

    # load in dataset
    dataloc = os.path.join("./data", f"{dataset}.npz")
    assert os.path.exists(dataloc)
    data = np.load(dataloc, allow_pickle=True)
    X, y = data['X'], data['y']

    dataset_resultsdir = os.path.join(args.resultsdir, dataset)
    if not os.path.exists(dataset_resultsdir):
        os.makedirs(dataset_resultsdir)
    timing_fname = os.path.join(dataset_resultsdir, "timing.csv")
    if os.path.exists(timing_fname):
        timing_df = pd.read_csv(timing_fname)
    else:
        timing_df = pd.DataFrame(columns=["name", "time_to_compute_P", "run_time_after_compute_P"])

    print("=====================================================")
    print(f"\tRunning experiments for {dataset} with values:")
    print(f"\tm (target dimension) = {m}, perplexity = {perplexity}, approx_nn = {approx_nn}, lr = {learning_rate}")
    print(f"\tknn_graph = {knn_graph_vals}")
    print(f"\tk_eigen = {k_eigen_vals}")
    print(f"\trepulsion_kernel = {rep_kernel_vals}, num_landmarks = {num_lm_vals}")
    print(f"\that_bandwidth = {hat_bandwidth_vals}")
    print(f"\trepulsion_rel_weight = {rep_rel_weights}")
    print("=====================================================\n")

    # iterate through each setting of graph hyperparameters
    for it, (knn_graph, k_eigen) in enumerate(graph_setting_list):
        print(f"Running test settings for (knn_graph, k_eigen) = {(knn_graph, k_eigen)}....")
        Lap_TSNE = LaplacianTSNE(n_components=m, knn_graph=knn_graph, perplexity=perplexity, k_eigen=k_eigen, \
                                 approx_nn=approx_nn, learning_rate=learning_rate, debug=args.debug)
        Lap_TSNE._prep_graph(X)

        # iterate through optimization hyperparameter settings to try
        for rep_rel_weight, repulsion_kernel, num_landmarks, hat_bandwidth in tqdm(opt_hyperparam_list, total=len(opt_hyperparam_list), \
                                                                   desc=f"Running tests for graph setting {it+1}/{len(graph_setting_list)}"):
            exp_name = f"{dataset}_{m}_{perplexity}_{str(approx_nn)}_{knn_graph}_{k_eigen}_{rep_rel_weight}_{repulsion_kernel}_{num_landmarks}_{learning_rate}_{hat_bandwidth}"
            
            # check if experiment already done (recorded in timing_df)
            if exp_name in list(timing_df.name.values):
                if not os.path.exists(os.path.join(args.resultsdir, f"{exp_name}.npy")):
                    print(f"\tNote: already found {exp_name} in timing.csv, but not its embedding... recomputing")
                else:
                    print(f"\tNote: Found already completed experiment, {exp_name}. Skipping...")
                    continue
            
            # run experiment
            X_embedded, run_time = run_experiment(X, LapTSNE=Lap_TSNE, repulsion_rel_weight=rep_rel_weight, \
                                                  repulsion_kernel=repulsion_kernel, num_landmarks=num_landmarks, \
                                                    hat_bandwidth=hat_bandwidth)
            
            # save embedding and results
            np.save(os.path.join(args.resultsdir, dataset, f"{exp_name}.npy"), X_embedded)
            timing_df.loc[len(timing_df) + 1] = [exp_name, Lap_TSNE.time_to_compute_P, run_time]
            timing_df.to_csv(timing_fname, index=None) # save at intermediate steps
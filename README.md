# laplacian-tsne
Faster implementation on t-SNE embeddings leveraging a Laplacian-based optimization framework. 

### Running tests in this repo

1. Make sure proper libraries are installed. For example, can create a conda environment from the given ``tsne.yml`` file provided:
```
conda env create -n tsne -f tsne.yml
conda activate tsne
```
2. Now, load datasets with ``python load_datasets.py``. This creates a directory with datasets for our experiments. 
3. Create config files like the provided ``config.yaml`` file. This specifies experiment hyperparameters to try on a given dataset. Check out ``config.yaml`` to see what values are currently implemented for changing in tests.
4. Run tests with ``python run.py --config config.yaml``, replacing ``config.yaml`` with the path to your desired config file. This will run experiments for all the given hyperparameter settings. 
    * Results are saved in the corresponding "./results/<DATASETNAME>" directory: the embedding ``X_`` as ``.npy`` file and timing results saved in ``timing.csv``
5. Run evaluations of embeddings with ``python evaluate.py``. Currently, only the knn Classifier accuracy is implemented, but we can add more. Based off code from the Rudin et al (2020) paper's GitHub, ``https://github.com/YingfanWang/PaCMAP/blob/master/evaluation/evaluation.py`` 
    * Results are saved in the same directory as in the previous step, but now in a file entitled ``eval.csv``
6. You can visualize the embeddings in the Jupyter notebook ``VisualizeEmbeddings.ipynb``. 


### Explanation of Hyperparameters:

* ``knn_graph``: # of k nearest neighbors in the graph we compute
* ``k_eigen``: # of eigenpairs to compute for the matrix $P$ corresponding to the computed graph
* ``approx_nn``: Whether or not to compute the knn graph via ``annoy`` package (approximate nearest neighbors) or "brute force" with sklearn's implementation with KDTree.
* ``repulsion_kernel``: Which repulsion kernel to use. Currently implemented options are: ``"hat"`` and ``"standard"``
* ``m``: target embedding dimension for tSNE to compute. 
* ``num_landmarks``: # of landmark points to use for approximating the repulsion kernel term in the objective function
* ``learning_rate``: learning rate value for the SGD with momentum optimization iterations
* ``hat_bandwidth``: bandwidth for the ``hat`` repulsion kernel.

In the config file, currently you specify a list of values to try for only the following parameters: ``knn_graph``, ``k_eigen``, ``repulsion_kernel``, ``num_landmarks``, and ``hat_bandwidth``.



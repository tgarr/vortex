# Construct the dataset

## Gist dataset

### 1. Download the dataset
Download the gist dataset could simply be done via running the download script we provided

```./download_gist.sh```

This script creates a gist folder that contains the data files: gist_base.fvecs,  gist_groundtruth.ivecs, gist_learn.fvecs, gist_query.fvecs

### 2. Clustering
There are two ways to build clusters, one is use FAISS KNN algorithm which could generate specified number of clusters, another way is to use balanced_knn algorithm from paper https://arxiv.org/abs/2403.01797 by Lars Gottesbüren, et al. We imported their codebase with customization for better alignment with our setup script

#### 2.1 FAISS KNN clustering
To use FAISS KNN clustering algorithm, you can simple run

``` python format_gist.py --embeddings_loc /path/to/save/embeddings --ncentroids 5 --niter 20```

#### 2.2 Use balanced KNN clustering
To get balanced knn clustering datasets, first needs to compile and build the gp-ann repository. It could be built either via cmake in that directory or the python file 
``` python build_balanced_gpann.py ```

To run gp-ann to generate balanced clustering, you can run with -b flag

``` python format_gist.py -b --embeddings_loc /path/to/save/embeddings --ncentroids 5 --gp_ann_loc ./gp-ann```
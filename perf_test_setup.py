#!/usr/bin/env python3

import numpy as np
import os
import pickle
import sys
import time
from derecho.cascade.external_client import ServiceClientAPI

from perf_config import *

np.random.seed(1234)             # make reproducible

OBJECT_POOLS_LIST = "setup/object_pools.list"

SUBGROUP_TYPES = {
        "VCSS": "VolatileCascadeStoreWithStringKey",
        "PCSS": "PersistentCascadeStoreWithStringKey",
        "TCSS": "TriggerCascadeNoStoreWithStringKey"
        }


def create_object_pool(capi, basepath):
    # create object pools
    print("Creating object pools ...")
    fpath = os.path.join(basepath,OBJECT_POOLS_LIST)
    with open(fpath,"r") as list_file:
        for line in list_file:
            fields = line.strip().split(" ")
            pool_path = fields[0]
            subgroup_type = fields[1]
            subgroup_index = int(fields[2])
            if len(fields) >= 4:
                affinity_set_regex = fields[3].strip()
                print(f"AFFINITY: {pool_path} {affinity_set_regex}")
                res = capi.create_object_pool(pool_path,SUBGROUP_TYPES[subgroup_type],
                                              subgroup_index,
                                              affinity_set_regex=affinity_set_regex)
                if res:
                    res.get_result()
                else:
                    print(f"Failed to create the object pool: {pool_path}")
                    exit(1)
            else:
                print(f"  {pool_path} {subgroup_type} {subgroup_index}")
                res = capi.create_object_pool(pool_path,
                                              SUBGROUP_TYPES[subgroup_type],
                                              subgroup_index)
                if res:
                    res.get_result()
                else:
                    print(f"Failed to create the object pool: {pool_path}")
                    exit(1)


def get_embeddings(basepath, filename, d=1024):
    '''
    Load the embeddings from a pickle file.
    '''
    fpath = os.path.join(basepath, filename)
    with open(fpath, 'rb') as file:
        data = pickle.load(file)
    assert(data.shape[1] == d)
    return data


def break_into_chunks(num_embeddings, chunk_size):
    chunk_idxs = []
    num_chunks = num_embeddings // chunk_size + 1 if num_embeddings % chunk_size != 0 else num_embeddings // chunk_size
    remaining_embs_num = num_embeddings
    for i in range(num_chunks):
        chunk_size = min(chunk_size, remaining_embs_num)
        start_idx = i*chunk_size
        end_idx = i*chunk_size + chunk_size
        chunk_idxs.append((start_idx, end_idx))
        remaining_embs_num -= chunk_size
    return chunk_idxs

    
def put_initial_embeddings(capi, basepath):
    print("Putting centroids and clusters' embeddings to cascade server ...")
    # 1. Put centroids'embeddings to cascade.
    centroid_file_name = 'validate_first_100_centroids.pickle'
    centroids_chunk_idx = break_into_chunks(NUM_CENTROIDS, NUM_EMB_PER_OBJ)
    print(f"NUM_CENTROIDS is {NUM_CENTROIDS}")
    centroids_embs = get_embeddings(basepath, centroid_file_name, EMBEDDING_DIM)
    for i, (start_idx, end_idx) in enumerate(centroids_chunk_idx):
        key = f"/rag/emb/centroids/{i}"
        centroids_embs_chunk = centroids_embs[start_idx:end_idx]
        res = capi.put(key, centroids_embs_chunk.tobytes())
        if res:
            res.get_result()
            print(f"Put the centroids embeddings {start_idx}:{end_idx} to key: {key}")
        else:
            print(f"Failed to put the centroids embeddings to key: {key}")
            exit(1)
    print("Initialized: Put the centroids embeddings")

    # 2. Put clusters' embeddings to cascade.
    centroid_count = centroids_embs.shape[0]
    cluster_file_name_list = [f'validate_first_100_ebd_doc_{count}.pickle' for count in range(centroid_count)]
    for cluster_id, cluster_file_name in enumerate(cluster_file_name_list):
        cluster_embs = get_embeddings(basepath, cluster_file_name, EMBEDDING_DIM)
        cluster_chunk_idx = break_into_chunks(NUM_EMB_PER_CENTROIDS, NUM_EMB_PER_OBJ)
        for i, (start_idx, end_idx) in enumerate(cluster_chunk_idx):
            key = f"/rag/emb/cluster{cluster_id}/{i}"
            cluster_embs_chunk = cluster_embs[start_idx:end_idx]
            res = capi.put(key, cluster_embs_chunk.tobytes())
            if res:
                res.get_result()
            else:
                print(f"Failed to put the cluster embeddings to key: {key}")
                exit(1)
    print(f"Initialized: Put the clusters embeddings")


def main(argv):
    print("Connecting to Cascade service ...")
    capi = ServiceClientAPI()
    create_object_pool(capi, "./")
    put_initial_embeddings(capi, "./perf_data/miniset/")
    print("Done!")


if __name__ == "__main__":
    main(sys.argv)
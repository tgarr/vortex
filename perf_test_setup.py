#!/usr/bin/env python3

import numpy as np
import os
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
                res = capi.create_object_pool(pool_path,SUBGROUP_TYPES[subgroup_type],subgroup_index,affinity_set_regex=affinity_set_regex)
                if res:
                    res.get_result()
                else:
                    print(f"Failed to create the object pool: {pool_path}")
            else:
                print(f"  {pool_path} {subgroup_type} {subgroup_index}")
                res = capi.create_object_pool(pool_path,SUBGROUP_TYPES[subgroup_type],subgroup_index)
                if res:
                    res.get_result()
                else:
                    print(f"Failed to create the object pool: {pool_path}")
            # time.sleep(0.5)


def get_embeddings(basepath, filename="centroids.pkl", d=64, num_embs=100):
    '''
    Load the embeddings from pickle files
    TODO: this is a placeholder, need to replace with real data
    '''
    xb = np.random.random((num_embs, d)).astype('float32')
    xb[:, 0] += np.arange(num_embs) / 1000.
    return xb


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
    fpath = os.path.join(basepath,OBJECT_POOLS_LIST)
    # 1. Put centroids'embeddings to cascade
    centroids_chunk_idx = break_into_chunks(NUM_CENTROIDS, NUM_EMB_PER_OBJ)
    #  TODO: replace with real data
    centroids_embs = get_embeddings(basepath, d=EMBEDDING_DIM, num_embs=NUM_CENTROIDS)
    for i, (start_idx, end_idx) in enumerate(centroids_chunk_idx):
        key = f"/rag/emb/centroids/{i}"
        centroids_embs_chunk = centroids_embs[start_idx:end_idx]
        res = capi.put(key, centroids_embs_chunk.tobytes())
        if res:
            res.get_result()
            print(f"Put the centroids embeddings to key: {key}")
        else:
            print(f"Failed to put the centroids embeddings to key: {key}")

    
    print("Initialized: Put the centroids embeddings")

    # 2. put clusters' embeddings to cascade
    for cluster_id in range(NUM_CENTROIDS):
        cluster_embs = get_embeddings(basepath, d=EMBEDDING_DIM, num_embs=NUM_EMB_PER_CENTROIDS)
        cluster_chunk_idx = break_into_chunks(NUM_EMB_PER_CENTROIDS, NUM_EMB_PER_OBJ)
        for i, (start_idx, end_idx) in enumerate(cluster_chunk_idx):
            key = f"/rag/emb/cluster{cluster_id}/{i}"
            cluster_embs_chunk = cluster_embs[start_idx:end_idx]
            res = capi.put(key, cluster_embs_chunk.tobytes())
            if res:
                res.get_result()
            else:
                print(f"Failed to put the cluster embeddings to key: {key}")
    print(f"Initialized: Put the clusters embeddings")


def main(argv):

    print("Connecting to Cascade service ...")
    capi = ServiceClientAPI()
    #basepath = os.path.dirname(argv[0])
    basepath = "."

    create_object_pool(capi, basepath)

    put_initial_embeddings(capi, basepath)

    print("Done!")

if __name__ == "__main__":
    main(sys.argv)


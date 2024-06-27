#!/usr/bin/env python3

import numpy as np
import os
import sys
import time
from derecho.cascade.external_client import ServiceClientAPI


OBJECT_POOLS_LIST = "setup/object_pools.list"

SUBGROUP_TYPES = {
        "VCSS": "VolatileCascadeStoreWithStringKey",
        "PCSS": "PersistentCascadeStoreWithStringKey",
        "TCSS": "TriggerCascadeNoStoreWithStringKey"
        }

EMBEDDING_DIM = 1024


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
                print(f"  {pool_path} {subgroup_type} {subgroup_index}")
                res = capi.create_object_pool(pool_path,SUBGROUP_TYPES[subgroup_type],subgroup_index)
                if res:
                    res.get_result()
            # time.sleep(0.5)


def get_embeddings(basepath, filename="centroids.pkl", d=64, num_embs=100):
    '''
    Load the embeddings from pickle files
    TODO: this is a placeholder, need to replace with real data
    '''
    np.random.seed(1234)             # make reproducible
    xb = np.random.random((num_embs, d)).astype('float32')
    xb[:, 0] += np.arange(num_embs) / 1000.
    return xb


def put_initial_embeddings(capi, basepath):
    print("Putting centroids and clusters' embeddings to cascade server ...")
    fpath = os.path.join(basepath,OBJECT_POOLS_LIST)
    num_clusters = 5 # TODO: temp start with 5 clusters
    # 1. Put centroids'embeddings to cascade
    centroids_embs = get_embeddings(basepath, filename="centroids.pkl", d=EMBEDDING_DIM, num_embs=num_clusters) 
    key = "/rag/emb/centroid_chunk0"
    capi.put(key, centroids_embs.tobytes())

    # 2. put clusters' embeddings to cascade
    for cluster_id in range(num_clusters):
        key = f"/rag/emb/cluster{cluster_id}"
        cluster_embs = get_embeddings(basepath, filename=f"cluster{cluster_id}.pkl", d=EMBEDDING_DIM, num_embs=100)
        capi.put(key, cluster_embs.tobytes())


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


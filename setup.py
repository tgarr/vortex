#!/usr/bin/env python3

import numpy as np
import os
import sys
import time
from derecho.cascade.external_client import ServiceClientAPI
sys.path.append(os.path.join(os.path.dirname(__file__), 'setup'))
from util import *

from perf_config import *

np.random.seed(1234)             # make reproducible

OBJECT_POOLS_LIST = "setup/object_pools.list"

SUBGROUP_TYPES = {
        "VCSS": "VolatileCascadeStoreWithStringKey",
        "PCSS": "PersistentCascadeStoreWithStringKey",
        "TCSS": "TriggerCascadeNoStoreWithStringKey"
        }

# EMBEDDING_DIM = 1024
NUM_CENTROIDS = 3 # TODO: temp start with 3 clusters
NUM_EMB_PER_CENTROIDS = 100


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


    

def put_initial_embeddings(capi, basepath):
    print("Putting centroids and clusters' embeddings to cascade server ...")
    fpath = os.path.join(basepath,OBJECT_POOLS_LIST)
    num_clusters = NUM_CENTROIDS
    # 1. Put centroids'embeddings to cascade
    
    key = f"/rag/emb/centroids_obj/0"
    centroids_embs = generate_random_embeddings(d=EMBEDDING_DIM, num_embs=num_clusters)
    res = capi.put(key, centroids_embs.tobytes())
    if res:
        res.get_result()
        print(f"Put the centroids embeddings to key: {key}")
    else:
        print(f"Failed to put the centroids embeddings to key: {key}")

    print("Initialized: Put the centroids embeddings")

    # 2. put clusters' embeddings to cascade
    for cluster_id in range(num_clusters):
        num_to_put = NUM_EMB_PER_CENTROIDS  
        for i in range(int(NUM_EMB_PER_CENTROIDS/NUM_EMB_PER_OBJ) + 1):
            key = f"/rag/emb/cluster{cluster_id}/{i}"
            obj_num_to_put = min(num_to_put, NUM_EMB_PER_OBJ)
            cluster_embs = generate_random_embeddings(d=EMBEDDING_DIM, num_embs=obj_num_to_put)
            res = capi.put(key, cluster_embs.tobytes())
            if res:
                res.get_result()
            else:
                print(f"Failed to put the cluster embeddings to key: {key}")
            num_to_put -= NUM_EMB_PER_OBJ
            i += 1
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


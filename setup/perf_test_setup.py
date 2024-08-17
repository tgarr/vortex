#!/usr/bin/env python3

import numpy as np
import os
import pickle
import sys
import time
import json
from derecho.cascade.external_client import ServiceClientAPI


NUM_EMB_PER_OBJ = 200  # < 1MB/4KB = 250

np.random.seed(1234)             # make reproducible
script_dir = os.path.dirname(os.path.abspath(__file__))

OBJECT_POOLS_LIST = "object_pools.list"

SUBGROUP_TYPES = {
        "VCSS": "VolatileCascadeStoreWithStringKey",
        "PCSS": "PersistentCascadeStoreWithStringKey",
        "TCSS": "TriggerCascadeNoStoreWithStringKey"
        }

ANSWER_MAPPING = {}


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

    
def put_initial_embeddings_docs(capi, basepath, doc_list_pathname = "doc_list.pickle"):
    print("Initialized: putting centroids and clusters' embeddings to cascade server ...")
    # 1. Put centroids'embeddings to cascade.
    centroid_file_name = 'validate_first_100_centroids.pickle'
    centroids_embs = get_embeddings(basepath, centroid_file_name, EMBEDDING_DIM)
    centroids_chunk_idx = break_into_chunks(centroids_embs.shape[0], NUM_EMB_PER_OBJ)
    print(f"number of centroids from pickle is {centroids_embs.shape[0]}")
    for i, (start_idx, end_idx) in enumerate(centroids_chunk_idx):
        key = f"/rag/emb/centroids_obj/{i}"
        centroids_embs_chunk = centroids_embs[start_idx:end_idx]
        res = capi.put(key, centroids_embs_chunk.tobytes())
        if res:
            res.get_result()
            print(f"Put the centroids embeddings to key: {key}, shape: {centroids_embs_chunk.shape}")
        else:
            print(f"Failed to put the centroids embeddings to key: {key}")
            exit(1)

    doc_list = pickle.load(open(os.path.join(basepath, doc_list_pathname), "rb"))
    global ANSWER_MAPPING
    # 2. Put clusters' embeddings and docs to cascade.
    centroid_count = centroids_embs.shape[0]
    cluster_file_name_list = [f'validate_first_100_ebd_doc_{count}.pickle' for count in range(centroid_count)]
    for cluster_id, cluster_file_name in enumerate(cluster_file_name_list):
        cluster_embs = get_embeddings(basepath, cluster_file_name, EMBEDDING_DIM)
        num_embeddings = cluster_embs.shape[0]
        cluster_chunk_idx = break_into_chunks(num_embeddings, NUM_EMB_PER_OBJ)
        for i, (start_idx, end_idx) in enumerate(cluster_chunk_idx):
            key = f"/rag/emb/cluster{cluster_id}/{i}"
            cluster_embs_chunk = cluster_embs[start_idx:end_idx]
            res = capi.put(key, cluster_embs_chunk.tobytes())
            if res:
                res.get_result()
                print(f"Put the cluster embeddings to key: {key}, shape: {cluster_embs_chunk.shape}")
            else:
                print(f"Failed to put the cluster embeddings to key: {key}")
                exit(1)
        # Put the corresponding docs to cascade
        for emb_idx in range(num_embeddings):
            doc_idx = ANSWER_MAPPING[cluster_id][emb_idx]
            doc_key = f"/rag/doc/{doc_idx}"
            doc = doc_list[doc_idx]
            res = capi.put(doc_key, doc.encode('utf-8'))
            if res:
                res.get_result()
            else:
                print(f"Failed to put the doc to key: {doc_key}")
                exit(1)
        print(f"Put cluster{cluster_id} docs to cascade")
    print(f"Initialized embeddings")


def put_doc_tables(capi, basepath, file_name):
    global ANSWER_MAPPING
    ANSWER_MAPPING = pickle.load(open(os.path.join(basepath, file_name), "rb"))
    print("Initialized: putting doc tables to cascade server ...")
    for i, (cluster_id, table_dict) in enumerate(ANSWER_MAPPING.items()):
        key = f"/rag/doc/emb_doc_map/cluster{cluster_id}"
        table_json = json.dumps(table_dict)
        res = capi.put(key, table_json.encode('utf-8'))
        if res:
            res.get_result()
        else:
            print(f"Failed to put the doc table to key: {key}")
            exit(1)
    print(f"Initialized tables")


def main(argv):
    print("Connecting to Cascade service ...")
    capi = ServiceClientAPI()
    create_object_pool(capi, script_dir)
    data_dir = os.path.join(script_dir, "perf_data/miniset/")
    put_doc_tables(capi, data_dir, "answer_mapping.pickle")
    put_initial_embeddings_docs(capi, data_dir,doc_list_pathname = "doc_list.pickle")
    print("Done!")


if __name__ == "__main__":
    main(sys.argv)
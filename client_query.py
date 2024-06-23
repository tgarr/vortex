#!/usr/bin/env python3

import json
import os
import sys
import time
from derecho.cascade.external_client import ServiceClientAPI
import numpy as np


OBJECT_POOLS_LIST = "setup/object_pools.list"

SUBGROUP_TYPES = {
        "VCSS": "VolatileCascadeStoreWithStringKey",
        "PCSS": "PersistentCascadeStoreWithStringKey",
        "TCSS": "TriggerCascadeNoStoreWithStringKey"
        }


def main(argv):

     print("Connecting to Cascade service ...")
     capi = ServiceClientAPI()
     #basepath = os.path.dirname(argv[0])
     basepath = "."

     # create object pools
     print("Creating object pools ...")
     capi.create_object_pool("/test", "VolatileCascadeStoreWithStringKey", 0)

     # array = np.array([1.1, 2.22, 3.333, 4.4444, 5.55555], dtype=np.float32)
     key = "/rag/emb/py_centroids_search/cluster0"
     query_list = ["hello world", "I am RAG"]
     json_string = json.dumps(query_list)
     encoded_bytes = json_string.encode('utf-8')
     capi.put("/rag/emb/py_centroids_search/cluster0", encoded_bytes)
     # capi.put("/test/hello", array.tostring())  # deprecated
     print(f"Put key:{key} value:{query_list} to Cascade.")

     print("Done!")

if __name__ == "__main__":
     main(sys.argv)


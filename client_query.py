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

MAX_RESULT_WAIT_TIME = 10 # seconds
RETRIEVE_WAIT_INTERVAL = 0.5 # seconds



def main(argv):

     print("Connecting to Cascade service ...")
     capi = ServiceClientAPI()
     #basepath = os.path.dirname(argv[0])
     basepath = "."

     # create object pools
     print("Creating object pools ...")
     capi.create_object_pool("/test", "VolatileCascadeStoreWithStringKey", 0)

     # array = np.array([1.1, 2.22, 3.333, 4.4444, 5.55555], dtype=np.float32)
     client_id = capi.get_my_id()
     querybatch_id = 0
     key = f"/rag/emb/py_centroids_search/client{client_id}qb{querybatch_id}"
     query_list = ["hello world", "I am RAG"]
     json_string = json.dumps(query_list)
     encoded_bytes = json_string.encode('utf-8')
     capi.put(key, encoded_bytes)
     # capi.put("/test/hello", array.tostring())  # deprecated
     print(f"Put key:{key} \n    value:{query_list} to Cascade.")
     result_key = "/rag/generate/" + f"client{client_id}qb{querybatch_id}_results"
     result_generated = False
     wait_time = 0
     while not result_generated and wait_time < MAX_RESULT_WAIT_TIME:
          result_future = capi.get(result_key)
          if result_future:
               res_dict = result_future.get_result()
               if len(res_dict['value']) > 0:
                    result_generated = True
                    print(f"Got result from key:{result_key} \n    value:{res_dict}")
          else:
               print(f"Getting key:{result_key} with NULL result_future.")
          time.sleep(RETRIEVE_WAIT_INTERVAL)
          wait_time += RETRIEVE_WAIT_INTERVAL

     print("Done!")

if __name__ == "__main__":
     main(sys.argv)


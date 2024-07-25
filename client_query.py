#!/usr/bin/env python3

import json
import os
import sys
import time
from derecho.cascade.external_client import ServiceClientAPI
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'setup'))
from util import *

from perf_config import *


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

     # array = np.array([1.1, 2.22, 3.333, 4.4444, 5.55555], dtype=np.float32)
     client_id = capi.get_my_id()
     querybatch_id = 0
     key = f"/rag/emb/centroids_search/client{client_id}qb{querybatch_id}"
     query_list = ["hello world", "I am RAG"]
     retrieved_queries = set() # track the queries to be retrieved
     query_list_json = json.dumps(query_list)
     query_list_json_bytes = query_list_json.encode('utf-8')
     emb_list = generate_random_embeddings(d=EMBEDDING_DIM, num_embs=len(query_list))
     emb_list_bytes = emb_list.tobytes()
     num_queries = len(query_list)
     num_queries_bytes = num_queries.to_bytes(4, byteorder='big', signed=False)
     # TODO: add comments
     query_embeddings_and_query_list = num_queries_bytes + emb_list_bytes + query_list_json_bytes
     capi.put(key, query_embeddings_and_query_list)
     # capi.put("/test/hello", array.tostring())  # deprecated
     print(f"Put key:{key} to Cascade.")
     result_prefix = "/rag/generate/" + f"client{client_id}qb{querybatch_id}_result"
     result_generated = False
     wait_time = 0
     while not result_generated and wait_time < MAX_RESULT_WAIT_TIME:
          existing_queries = capi.list_keys_in_object_pool(result_prefix)
          new_keys_to_retrieve = [] 
          # Need to process all the futures, because embedding objects may hashed to different shards
          for r in existing_queries:
               keys = r.get_result()
               for key in keys:
                    if key not in retrieved_queries:
                         new_keys_to_retrieve.append(key)
                         retrieved_queries.add(key)
          for result_key in new_keys_to_retrieve:
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
     if not result_generated:
          print(f"Failed to get result from request:{key} after waiting for {MAX_RESULT_WAIT_TIME} seconds.")

if __name__ == "__main__":
     main(sys.argv)


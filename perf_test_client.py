#!/usr/bin/env python3

import json
import numpy as np
import os
import pickle
import sys
import time
from derecho.cascade.external_client import ServiceClientAPI
from derecho.cascade.external_client import TimestampLogger


from perf_config import *
import logging
logging.basicConfig(format='%(asctime)s %(name)16s[%(levelname)s] %(message)s')


OBJECT_POOLS_LIST = "setup/object_pools.list"


SUBGROUP_TYPES = {
        "VCSS": "VolatileCascadeStoreWithStringKey",
        "PCSS": "PersistentCascadeStoreWithStringKey",
        "TCSS": "TriggerCascadeNoStoreWithStringKey"
        }


QUERY_PICKLE_FILE = "validate_first_100_query.pickle"
BASE_PATH = "./perf_data/miniset/"
query_list = None


def get_queries(basepath, filename, batch_id):
     '''
     Load the queries from pickle files
     '''
     # queries = []
     # with open(os.path.join(basepath, filename), "rb") as f:
     #      queries = pickle.load(f)
     # return queries
     # query_list = ["How big is an apple?", 
     #                "What is the capital of France?", 
     #                "What is the weather in New York?",
     #                "How is the hotpot at Haidilao?",
     #                "What is the best way to cook a steak?",
     #                "Who is the most popular singer in the world?",
     #                "How to make a cake?",
     #                "How to make a cocktail?",
     #                "What is the best way to play chess?",
     #                "How to play basketball?"]
     # return query_list[:batch_size]
     global query_list
     if query_list is None:
          fpath = os.path.join(basepath, filename)
          with open(fpath, 'rb') as file:
               query_list = pickle.load(file)
     return query_list[QUERY_PER_BATCH * batch_id : QUERY_PER_BATCH * (batch_id + 1)]


collected_all_results = True # TODO: check in code whether all queries have received results

def main(argv):
     capi = ServiceClientAPI()
     print("Connected to Cascade service ...")
     client_id = capi.get_my_id()
     tl = TimestampLogger()

     # for querybatch_id in range(TOTAL_BATCH_COUNT):
     for querybatch_id in range(1):
          # Send batch of queries to Cascade service
          key = f"/rag/emb/py_centroids_search/client{client_id}qb{querybatch_id}"
          query_list = get_queries(BASE_PATH, QUERY_PICKLE_FILE, querybatch_id)
          json_string = json.dumps(query_list)
          encoded_bytes = json_string.encode('utf-8')
          tl.log(LOG_TAG_QUERIES_SENDING_START, client_id, querybatch_id, 0)
          capi.put(key, encoded_bytes)
          tl.log(LOG_TAG_QUERIES_SENDING_END, client_id, querybatch_id, 0)
          if PRINT_DEBUG_MESSAGE:
               print(f"Put queries to key:{key}, batch_size:{len(query_list)}")

          # Wait for result of this batch
          result_key = "/rag/generate/" + f"client{client_id}qb{querybatch_id}_results"
          result_generated = False
          wait_time = 0
          while not result_generated and wait_time < MAX_RESULT_WAIT_TIME:
               result_future = capi.get(result_key)
               if result_future:
                    res_dict = result_future.get_result()
                    if len(res_dict['value']) > 0:
                         result_generated = True
                         tl.log(LOG_TAG_QUERIES_RESULT_CLIENT_RECEIVED,client_id,querybatch_id,0)
                         if True:
                              # result dictionary format["query_id": (float(distance), int(cluster_id), int(emb_id)), "query_id":(...) ... ]
                              print(f"Got result from key:{result_key}, value:{res_dict['value']}")
               else:
                    if PRINT_DEBUG_MESSAGE:
                         print(f"Getting key:{result_key} with NULL result_future.")
               time.sleep(RETRIEVE_WAIT_INTERVAL)
               wait_time += RETRIEVE_WAIT_INTERVAL

     tl.flush(f"client_timestamp.dat")

     # notify all nodes to flush logs
     # TODO: read it from object_pool.list to send this notification to both /rag/emb and /rag/generate object pool's subgroup's shards
     subgroup_type = SUBGROUP_TYPES["VCSS"]
     subgroup_index = 0
     num_shards = len(capi.get_subgroup_members(subgroup_type,subgroup_index))
     for i in range(num_shards):
          shard_index = i
          capi.put("/rag/flush/notify", b"",subgroup_type=subgroup_type,subgroup_index=subgroup_index,shard_index=shard_index)

     print("Done!")


if __name__ == "__main__":
     main(sys.argv)

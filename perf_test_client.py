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


def get_queries(basepath, filename, query_index_range):
     """Load the queries from pickle file.

     Args:
          basepath: The path of directory where the query pickle file resides.
          filename: The filename of the query pickle file.
          query_index_range: A pair of (start_index, end_index), which is non-inclusive.
     
     Returns:
          The queries in plain texts that fall in the query index range.

     Raises:
          IndexError: This is raised when the query_index_range is larger than the range of queries we have.
     """
     global query_list
     if query_list is None:
          fpath = os.path.join(basepath, filename)
          with open(fpath, 'rb') as file:
               query_list = pickle.load(file)
     return query_list[query_index_range[0] : query_index_range[1]]


collected_all_results = True # TODO: check in code whether all queries have received results


def get_query_batch():
     """Get the start and end index of each query batch following Poisson distribution.

     Returns:
          [(start_index, end_index), ...]
     """
     query_intervals = np.random.Generator.exponential(1/AVG_QUERY_PER_BATCH, 
                                                       int(AVG_QUERY_PER_BATCH * TOTAL_BATCH_COUNT * 1.5))
     cumulated_intervals = np.cumsum(query_intervals)
     query_count_list = list([0])
     for batch in range(TOTAL_BATCH_COUNT):
          previous_total = cumulated_intervals[cumulated_intervals <= batch].shape[0]
          current_total = cumulated_intervals[cumulated_intervals <= batch + 1].shape[0]
          query_count_list.append(current_total - previous_total)
     query_count_array = np.cumsum(np.array(query_count_list, dtype=int))
     res_query_count_list = [(query_count_array[i - 1], query_count_array[i]) for i in range(1, TOTAL_BATCH_COUNT + 1)]
     return res_query_count_list


def main(argv):
     capi = ServiceClientAPI()
     print("Connected to Cascade service ...")
     client_id = capi.get_my_id()
     tl = TimestampLogger()
     query_batch_list = get_query_batch()

     for querybatch_id in range(TOTAL_BATCH_COUNT):
          # Send batch of queries to Cascade service
          key = f"/rag/emb/py_centroids_search/client{client_id}qb{querybatch_id}"
          query_list = get_queries(BASE_PATH, QUERY_PICKLE_FILE, query_batch_list[querybatch_id])
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
                         if PRINT_DEBUG_MESSAGE == 1:
                              # result dictionary format["query_id": (float(distance), int(cluster_id), int(emb_id)), "query_id":(...) ... ]
                              print(f"Got result from key:{result_key}, value:{res_dict['value']}")
               else:
                    if PRINT_DEBUG_MESSAGE == 1:
                         print(f"Getting key:{result_key} with NULL result_future.")
               time.sleep(RETRIEVE_WAIT_INTERVAL)
               wait_time += RETRIEVE_WAIT_INTERVAL
          if not result_generated:
               print(f"Failed to get result for querybatch_id:{querybatch_id} after {MAX_RESULT_WAIT_TIME} seconds.")
          if (querybatch_id + 1) % PRINT_FINISH_INTEVAL == 0:
               print(f"Finished processing query_batch {querybatch_id}")

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
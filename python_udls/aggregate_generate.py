#!/usr/bin/env python3
import cascade_context
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger
from derecho.cascade.udl import UserDefinedLogic
from collections import defaultdict
import io
import numpy as np
import json
import re
import time


class ClusterSearchResults:
     def __init__(self, cluster_counts, top_k):
          '''
          Constructor
          @param cluster_counts: the total number of clusters this query is searching
          @param top_k: the number of top results to return
          '''
          self.cluster_counts = cluster_counts  
          self.top_k = top_k
          # formated as {cluster0:{"emb_id1": distance1, ...}, cluster1:{"emb_id1": distance1, ...}, ...}
          self.cluster_results = {}
          self.collected_all_results = False  # true if all the selected clusters results are collected
          self.agg_top_k_res = None
          
     def select_top_K(self):
          '''
          Select the top top_k results from all cluster results
          @return a list of top_K cluster_id and emb_id, ordered from closest to farthest
          '''
          all_results = []
          for cluster_id, embeddings in self.cluster_results.items():
               for emb_id, distance in embeddings.items():
                    all_results.append((distance, cluster_id, emb_id))

          all_results.sort(key=lambda x: x[0])
          top_k_results = all_results[:self.top_k]
          return top_k_results


     def add_cluster_result(self, cluster_id, cluster_search_results):
          self.cluster_results[cluster_id] = cluster_search_results
          if len(self.cluster_results) == self.cluster_counts:
               self.agg_top_k_res = self.select_top_K()
               self.collected_all_results = True



     
          
     



# TODO: this implementation has a lot of copies for small objects (query results), 
#       could be optimized if implemented in C++. But may not be the bottleneck
class AggregateGenerateUDL(UserDefinedLogic):
     def __init__(self,conf_str):
          '''
          Constructor
          '''
          # collect the cluster search result {(query_batch_key,query_count):{query_id: ClusterSearchResults, ...}, ...}
          self.cluster_search_res = {}
          self.conf = json.loads(conf_str)
          self.top_k = int(self.conf["top_k"])
          self.top_clusters_count = int(self.conf["top_clusters_count"])
          self.capi = ServiceClientAPI()
          
          

     def check_client_batch_finished(self, query_batch_key, query_count):
          '''
          Check if all queries in the client batch are finished
          @param query_batch_key: the identification key for this batch of queries from a client
          @return True if all queries in the client batch are finished
          '''
          collected_all_results = False
          if (query_batch_key, query_count) in self.cluster_search_res:
               if len(self.cluster_search_res[(query_batch_key, query_count)]) == query_count:
                    collected_all_results = all([self.cluster_search_res[(query_batch_key, query_count)][qid].collected_all_results for qid in range(query_count)])
          return collected_all_results

     def format_client_batch_result(self, query_batch_key, query_count):
          '''
          Format the client batch result for the next UDL
          @param query_batch_key: the identification key for this batch of queries from a client
          @param query_count: the number of queries in this batch
          @return the formatted client batch result
          '''
          client_query_batch_result = {}
          for qid in range(query_count):
               client_query_batch_result[qid] = self.cluster_search_res[(query_batch_key, query_count)][qid].agg_top_k_res
          return client_query_batch_result

     def ocdpo_handler(self,**kwargs):
          key = kwargs["key"]
          blob = kwargs["blob"]
          pathname = kwargs["pathname"]

          qc_index = key.find("_qc")
          if qc_index == -1:
               # This shouldn't happen becuase the attached "_qc" is done by encode_centroids_search_udl implemented by us
               print(f"[AggregateGenerate] received an object with invalid key format")
               return (key, "")
          query_batch_key = key[:qc_index]
          part_after = key[qc_index:]
          match = re.match(r"_qc(\d+)_cluster(\d+)_qid(\d+)", part_after)
          if not match:
               # This also shouldn't happen becuase the format is done by encode_centroids_search_udl, and clusters_search_udl implemented by us
               print(f"[AggregateGenerate] received an object with invalid key format")
               return
          query_count = int(match.group(1))  # number of queries in this client request batch
          cluster_id = int(match.group(2))
          qid = int(match.group(3))
          

          # 1. parse the blob to dict
          bytes_obj = blob.tobytes()
          json_str_decoded = bytes_obj.decode('utf-8')
          cluster_result = json.loads(json_str_decoded)
          query = cluster_result["query"]
          
          # 2. add the cluster result to the aggregated query results
          if (query_batch_key, query_count) not in self.cluster_search_res:
               self.cluster_search_res[(query_batch_key, query_count)] = {}
          if qid not in self.cluster_search_res[(query_batch_key, query_count)]:
               self.cluster_search_res[(query_batch_key, query_count)][qid] = ClusterSearchResults(self.top_clusters_count, self.top_k)
          self.cluster_search_res[(query_batch_key,query_count)][qid].add_cluster_result(cluster_id, cluster_result)
          if not self.cluster_search_res[(query_batch_key,query_count)][qid].collected_all_results:
               return

          # 3. check if all results for this batch of queries are collected
          if self.check_client_batch_finished(query_batch_key, query_count):
               # 3.1 format the key and value to be saved 
               next_key = f"/rag/generate/{query_batch_key}_results"
               client_query_batch_result = self.format_client_batch_result(query_batch_key, query_count)
               sorted_client_query_batch_result = {k: client_query_batch_result[k] for k in sorted(client_query_batch_result)}
               client_query_batch_result_json = json.dumps(sorted_client_query_batch_result)
               # 3.2 save the result as KV object in cascade to be retrieved by client
               self.capi.put(next_key, client_query_batch_result_json.encode('utf-8'))
               print(f"[AggregateGenerate] put the agg_results to key:{next_key},\
                         \n                   value: {sorted_client_query_batch_result}")


               

     def __del__(self):
          pass
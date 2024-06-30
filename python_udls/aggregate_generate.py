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
          

     def add_cluster_result(self, cluster_id, cluster_search_results):
          self.cluster_results[cluster_id] = cluster_search_results

     def collected_all_results(self):
          return len(self.cluster_results) == self.cluster_counts

     def select_top_K(self):
          '''
          Select the top top_k results from all cluster results
          @return top_K cluster_id and emb_id
          '''
          all_results = []
          for cluster_id, embeddings in self.cluster_results.items():
               for emb_id, distance in embeddings.items():
                    all_results.append((distance, cluster_id, emb_id))

          all_results.sort(key=lambda x: x[0])
          top_k_results = all_results[:self.top_k]
          return top_k_results
          





# TODO: this implementation has a lot of copies for small objects (query results), 
#       could be optimized if implemented in C++. But may not be the bottleneck
class AggregateGenerateUDL(UserDefinedLogic):
     def __init__(self,conf_str):
          '''
          Constructor
          '''
          # Aggregated query results {(client_id,query_id):{query_id: ClusterSearchResults, ...}, ...}
          self.agg_query_results = {}
          self.conf = json.loads(conf_str)
          self.top_k = int(self.conf["top_k"])
          self.top_clusters_count = int(self.conf["top_clusters_count"])
            


     def ocdpo_handler(self,**kwargs):
          key = kwargs["key"]
          blob = kwargs["blob"]
          pathname = kwargs["pathname"]
          match = re.match(r"client(\d+)qb(\d+)_cluster(\d+)_qid(\d+)", key)
          if not match:
               print(f"[AggregateGenerate] received an object with invalid key format")
               return
          client_id = int(match.group(1))
          querybatch_id = int(match.group(2))
          cluster_id = int(match.group(3))
          qid = int(match.group(4))
          

          # 1. parse the blob to dict
          bytes_obj = blob.tobytes()
          json_str_decoded = bytes_obj.decode('utf-8')
          cluster_result = json.loads(json_str_decoded)
          query = cluster_result["query"]
          print(f"----- query :{query}")
          
          # 2. add the cluster result to the aggregated query results
          if (client_id, querybatch_id) not in self.agg_query_results:
               self.agg_query_results[(client_id, querybatch_id)] = {}
          if qid not in self.agg_query_results[(client_id, querybatch_id)]:
               self.agg_query_results[(client_id, querybatch_id)][qid] = ClusterSearchResults(self.top_clusters_count, self.top_k)
          self.agg_query_results[(client_id, querybatch_id)][qid].add_cluster_result(cluster_id, cluster_result)
          
          # 3. check if all results are collected
          if self.agg_query_results[(client_id, querybatch_id)][qid].collected_all_results():
               # 4. aggregate the results
               top_k_results = self.agg_query_results[(client_id, querybatch_id)][qid].select_top_K()
               print(f"~~~~~~ [AggregateGenerate] client{client_id}batch{querybatch_id} \
                      \n       query: {query} \
                      \n       top_k_results: {top_k_results}")


               

     def __del__(self):
          pass
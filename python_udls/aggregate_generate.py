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




# TODO: this implementation has a lot of copies for small objects (query results), 
#       could be optimized if implemented in C++. But may not be the bottleneck
class AggregateGenerateUDL(UserDefinedLogic):
     def __init__(self,conf_str):
          '''
          Constructor
          '''
          # Aggregated query results {(client_id,query_id):{query_id: ClusterSearchResults, ...}, ...}
          self.agg_query_results = {}
            


     def ocdpo_handler(self,**kwargs):
          key = kwargs["key"]
          blob = kwargs["blob"]
          pathname = kwargs["pathname"]
          # message_id = kwargs["message_id"]
          # key format: "client{client_id}qb{querybatch_id}qid{qid}top{topK}_cluster{cluster_id}"
          # value format: json string of {emb_id: distance, ...}
          # extract client_id, querybatch_id, qid, topK, cluster_id
          match = re.match(r"client(\d+)qb(\d+)qid(\d+)top(\d+)_cluster(\d+)", key)
          if not match:
               print("[AggregateGenerate] invalid key format")
               return
          client_id = int(match.group(1))
          querybatch_id = int(match.group(2))
          qid = int(match.group(3))
          topK = int(match.group(4))
          cluster_id = int(match.group(5))

          # 1. parse the blob to dict
          bytes_obj = blob.tobytes()
          json_str_decoded = bytes_obj.decode('utf-8')
          cluster_result = json.loads(json_str_decoded)
          
          # 2. add the cluster result to the aggregated query results
          if (client_id, querybatch_id) not in self.agg_query_results:
               self.agg_query_results[(client_id, querybatch_id)] = {}
          if qid not in self.agg_query_results[(client_id, querybatch_id)]:
               # TODO: currently assume we want to select topK from topK cluster's results
               self.agg_query_results[(client_id, querybatch_id)][qid] = ClusterSearchResults(topK, topK)
          self.agg_query_results[(client_id, querybatch_id)][qid].add_cluster_result(cluster_id, cluster_result)
          
          # 3. check if all results are collected
          if self.agg_query_results[(client_id, querybatch_id)][qid].collected_all_results():
               print(f"~~~~~~ [AggregateGenerate] collected all results for client{client_id}qb{querybatch_id}qid{qid}")
               # 4. aggregate the results


               

     def __del__(self):
          pass
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
          # Aggregated query results {(client_id,query_batch_id,query_count):{query_id: ClusterSearchResults, ...}, ...}
          self.agg_query_results = {}
          # Aggregated client batch results {(client_id, querybatch_id,query_count): {query_id: top_k_results, ...}, ...}
          self.agg_client_batch_results = {}
          self.conf = json.loads(conf_str)
          self.top_k = int(self.conf["top_k"])
          self.top_clusters_count = int(self.conf["top_clusters_count"])
          self.capi = ServiceClientAPI()
          
          

     def check_client_batch_finished(self, client_id, querybatch_id, query_count):
          '''
          Check if all queries in the client batch are finished
          @param client_id: the client id
          @param querybatch_id: the query batch id
          @return True if all queries in the client batch are finished
          '''
          return (client_id, querybatch_id, query_count) in self.agg_client_batch_results and \
                 len(self.agg_client_batch_results[(client_id, querybatch_id, query_count)]) == query_count

     def ocdpo_handler(self,**kwargs):
          key = kwargs["key"]
          blob = kwargs["blob"]
          pathname = kwargs["pathname"]
          # TODO: rethink here, maybe shouldn't assume client key format to have client[client_id]qb[querybatch_id], 
          #       but should start from qc.
          match = re.match(r"client(\d+)qb(\d+)qc(\d+)_cluster(\d+)_qid(\d+)", key)
          if not match:
               print(f"[AggregateGenerate] received an object with invalid key format")
               return
          client_id = int(match.group(1))
          querybatch_id = int(match.group(2))
          query_count = int(match.group(3))  # number of queries in this client request batch
          cluster_id = int(match.group(4))
          qid = int(match.group(5))
          

          # 1. parse the blob to dict
          bytes_obj = blob.tobytes()
          json_str_decoded = bytes_obj.decode('utf-8')
          cluster_result = json.loads(json_str_decoded)
          query = cluster_result["query"]
          
          # 2. add the cluster result to the aggregated query results
          if (client_id, querybatch_id, query_count) not in self.agg_query_results:
               self.agg_query_results[(client_id, querybatch_id, query_count)] = {}
          if qid not in self.agg_query_results[(client_id, querybatch_id, query_count)]:
               self.agg_query_results[(client_id, querybatch_id, query_count)][qid] = ClusterSearchResults(self.top_clusters_count, self.top_k)
          self.agg_query_results[(client_id, querybatch_id,query_count)][qid].add_cluster_result(cluster_id, cluster_result)

          # 3. check if all results are collected
          if self.agg_query_results[(client_id, querybatch_id, query_count)][qid].collected_all_results():
               # 4. aggregate the results
               top_k_results = self.agg_query_results[(client_id, querybatch_id, query_count)][qid].select_top_K()
               # 5. put the aggregated results to the aggregated client batch results
               if (client_id, querybatch_id, query_count) not in self.agg_client_batch_results:
                    self.agg_client_batch_results[(client_id, querybatch_id, query_count)] = {}
               self.agg_client_batch_results[(client_id, querybatch_id, query_count)][qid] = top_k_results
               # 6. check if all queries in the client batch are finished
               if self.check_client_batch_finished(client_id, querybatch_id, query_count):
                    # 7. if all queries in the client batch are finished, put the aggregated client batch results to the next UDL
                    next_key = f"/rag/generate/client{client_id}qb{querybatch_id}_results"
                    client_query_batch_result = self.agg_client_batch_results[(client_id, querybatch_id, query_count)]
                    sorted_client_query_batch_result = {k: client_query_batch_result[k] for k in sorted(client_query_batch_result)}
                    client_query_batch_result_json = json.dumps(sorted_client_query_batch_result)
                    self.capi.put(next_key, client_query_batch_result_json.encode('utf-8'))
                    print(f"[AggregateGenerate] put the agg_results to key:{next_key},\
                          \n                   value: {sorted_client_query_batch_result}")


               

     def __del__(self):
          pass
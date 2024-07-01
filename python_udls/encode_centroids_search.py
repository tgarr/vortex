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
import warnings
warnings.filterwarnings("ignore")

from FlagEmbedding import BGEM3FlagModel
import faiss    




class EncodeCentroidsSearchUDL(UserDefinedLogic):
     def __init__(self,conf_str):
          '''
          Constructor
          '''
          self.encoder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, device="cpu")
          self.capi = ServiceClientAPI()
          self.centroids_embeddings = None

          self.num_embs = 0

          self.conf = json.loads(conf_str)
          self.top_k = int(self.conf["top_k"])
          self.emb_dim = int(self.conf["emb_dim"])
          self.index = faiss.IndexFlatL2(self.emb_dim)
     
     def get_centroid_obj_keys(self, capi):
          '''
          Get the sorted keys for embeddings
          Needs to be sorted, because the centroid embeddings are formed in the order of its corresponding cluster index
          @param capi: the ServiceClientAPI object
          @return the keys for embeddings
          '''
          centroids_key_prefix = "/rag/emb/centroids"
          res = capi.list_keys_in_object_pool(centroids_key_prefix)
          centroids_obj_keys = []
          # Need to process all the futures, because embedding objects may hashed to different shards
          for r in res:
               keys = r.get_result()
               for key in keys:
                    if len(key) > len(centroids_key_prefix):
                         centroids_obj_keys.append(key)
          unique_centroids_obj_keys = list(set(centroids_obj_keys))
          # +1 to account for / before object number, because keys are formated as "/rag/emb/centroids/0"
          sorted_centroids_obj_keys = sorted(unique_centroids_obj_keys, key=lambda x: int(x[len(centroids_key_prefix)+1:])) 
          return sorted_centroids_obj_keys

     def load_centroids_embeddings(self):
          '''
          Load the centroids embeddings
          Fill in the memory cache by getting the centroids embeddings from KV store in Cascade
          This function is called when this UDL is first triggered by caller to operator(),
          the embeddings for all centroids are used to compute centroids search and find the closest clusters.
          it sacrifices the first request to this node, but the following requests will benefit from this cache.
          The embeddings for centroids are stored as KV objects in Cascade.
          In static RAG setting, this function should be called only once at the begining.
          In dynamic RAG setting, this function could be extended to call periodically or upon notification. 
          (The reason not to call it at initialization, is that initialization is called upon server starts, 
          but the data have not been put to the servers yet, this needs to be called after the centroids data are put)
          '''
          centroids_obj_keys = self.get_centroid_obj_keys(self.capi)
          if len(centroids_obj_keys) == 0:
               print(f"Failed to get the centroids embeddings")
               return
          for cluster_key in centroids_obj_keys:
               res = self.capi.get(cluster_key)
               if not res:
                    print(f"Failed to get the centroids embeddings from key: {cluster_key}")
                    return
               emb = res.get_result()['value']
               flattend_emb = np.frombuffer(emb, dtype=np.float32)
               flattend_emb = flattend_emb.reshape(-1, self.emb_dim) # FAISS PYTHON API requires to reshape to 2D array
               if self.centroids_embeddings is None:
                    self.centroids_embeddings = flattend_emb
               else:
                    self.centroids_embeddings = np.concatenate((self.centroids_embeddings, flattend_emb), axis=0)
          print(f"loaded centroid_embeddings shape: {self.centroids_embeddings.shape}")
          self.index.add(self.centroids_embeddings)
          self.have_centroids_loaded = True
          # array1 = np.concatenate((array1, array2), axis=0)


     def combine_common_clusters(self, knn_result_I):
          '''
          combine queries that have the same top_k neighbor clusters results,
          to do batch message sending and processing
          @param knn_result_I: the top_k neighbor clusters for each query. 
                                Result from faiss knn index search
          '''
          neighbor_to_queries = defaultdict(list)
          for i, neighbors in enumerate(knn_result_I):
               for neighbor in neighbors:
                    neighbor_to_queries[neighbor].append(i)
          return neighbor_to_queries


     def ocdpo_handler(self,**kwargs):
          key = kwargs["key"]
          blob = kwargs["blob"]
          pathname = kwargs["pathname"]
          message_id = kwargs["message_id"]
          # 0. load centroids' embeddings
          if self.centroids_embeddings == None:
               self.load_centroids_embeddings()
          # 1. process the queries from blob to embeddings
          decoded_json_string = blob.tobytes().decode('utf-8')
          query_list = json.loads(decoded_json_string)
          print(query_list)
          encode_result = self.encoder.encode(
               query_list, return_dense=True, return_sparse=False, return_colbert_vecs=False
          )
          query_embeddings = encode_result['dense_vecs']
          query_embeddings = query_embeddings[:, :self.emb_dim]  # TODO: remove this. temporarily use this to test original faiss implementation
          print(f"shape of query embeddings: {query_embeddings.shape}")
          # 2. search the centroids
          D, I = self.index.search(query_embeddings, self.top_k)     # actual search
          print(I[:5])                   # print top 5 query top_k neighbors
          # 3. trigger the subsequent UDL by evict the query to the top M shards according to affinity set sharding policy
          clusters_to_queries = self.combine_common_clusters(I)
          nq = len(query_list)
          for cluster_id, query_ids in clusters_to_queries.items():
               print(f"cluster_id: {cluster_id}, query_ids: {query_ids}")
               if cluster_id == -1:
                    # print error message and stop the ocdpo_handler
                    print(f"Error: cluster_id is -1. Stopped processing of key({key}) at EncodeCentroidsSearchUDL.")
                    return
               # 3.1 construct new key for subsequent udl based on cluster_id and query_ids
               ''' 
               Current key_string is in the format of  "/rag/emb/centroids_search/client{client_id}qb{querybatch_id}"
               Change to format of "/rag/emb/centroids_search/client{client_id}qb{querybatch_id}qc{client_batch_query_count}_cluster{cluster_id}"
               '''
               key_string = f"{key}qc{nq}_cluster{cluster_id}"
               print(f"EncodeCentroidsSearchUDL: emitting subsequent for key({key_string})")
               # 3.2 construct new blob for subsequent udl based on query_ids
               query_embeddings_for_cluster = query_embeddings[query_ids]
               query_embeddings_bytes = query_embeddings_for_cluster.tobytes()
               # Note: constructed this way to keep the order of query_ids in query_dict
               query_dict = {query_id: query_list[query_id] for query_id in query_ids}
               query_list_json = json.dumps(query_dict)
               query_list_json_bytes = query_list_json.encode('utf-8')
               num_queries = len(query_ids)
               num_queries_bytes = num_queries.to_bytes(4, byteorder='big')
               query_embeddings_and_query_list =  num_queries_bytes + query_embeddings_bytes + query_list_json_bytes
               cascade_context.emit(key_string, query_embeddings_and_query_list, message_id=message_id)
          print(f"EncodeCentroidsSearchUDL: emitted subsequent for key({key})")

     def __del__(self):
          pass
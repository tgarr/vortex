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
          self.have_centroids_loaded = False

          self.top_K = 4  # TODO: hard code for now, use the number in faiss
          self.emb_dim = 64 # TODO: get this from config, for now hard code it
          self.num_embs = 0
          self.index = faiss.IndexFlatL2(self.emb_dim)
          
     

     def load_centroids_embeddings(self, cluster_id=0):
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
          self.centroids_embeddings = {}
          # TODO: here instead of assuming 1 centroid object, there could be multiple sentroid objects
          #       use list_keys
          cluster_key = f"/rag/emb/centroid_chunk0"
          res = self.capi.get(cluster_key)
          if not res:
               print(f"Failed to get the centroids embeddings from key: {cluster_key}")
               return
          emb = res.get_result()['value']
          flattend_emb = np.frombuffer(emb, dtype=np.float32)
          self.centroids_embeddings = flattend_emb.reshape(-1, self.emb_dim) # reshape to 2D array
          print(f"loaded centroid_embeddings shape: {self.centroids_embeddings.shape}")
          self.index.add(self.centroids_embeddings)
          self.have_centroids_loaded = True


     def combine_common_clusters(self, knn_result_I):
          '''
          combine queries that have the same top_K neighbor clusters results,
          to do batch message sending and processing
          @param knn_result_I: the top_K neighbor clusters for each query. 
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
          print("PYTHON Encode_centroids_search received key: ", key)
          # 0. load centroids' embeddings
          if not self.have_centroids_loaded:
               self.load_centroids_embeddings()
          # 1. process the queries from blob to embeddings
          decoded_json_string = blob.tobytes().decode('utf-8')
          query_list = json.loads(decoded_json_string)
          print(query_list)
          encode_result = self.encoder.encode(
               query_list, return_dense=True, return_sparse=False, return_colbert_vecs=False
          )
          query_embeddings = encode_result['dense_vecs']
          query_embeddings = query_embeddings[:, :64]  # TODO: remove this. temporarily use this to test original faiss implementation
          print(f"shape of query embeddings: {query_embeddings.shape}")
          # 2. search the centroids
          D, I = self.index.search(query_embeddings, self.top_K)     # actual search
          print(I[:5])                   # print top 5 query top_K neighbors
          # 3. trigger the subsequent UDL by evict the query to the top M shards according to affinity set sharding policy
          clusters_to_queries = self.combine_common_clusters(I)
          for cluster_id, query_ids in clusters_to_queries.items():
               print(f"cluster_id: {cluster_id}, query_ids: {query_ids}")
               # 3.1 construct new key for subsequent udl based on cluster_id and query_ids
               ''' 
               Current key_string is in the format of  "/rag/emb/centroids_search/client{client_id}qb{querybatch_id}"
               Change to format of "/rag/emb/centroids_search/client{client_id}qb{querybatch_id}{qids-topK}_cluster{cluster_id}"
               The last part {qid-topK} is a map from query_id to the top_K neighbor index in the cluster_id
                    e.g. {qid-topK}="qids0-2-5top4" indicate this object contains query embeddings for query 0, 2, 5
                    and they use the top 4 neighbors in the cluster_id
               '''
               key_string = f"{key}qids{'-'.join([str(qid) for qid in query_ids])}top{self.top_K}_cluster{cluster_id}"
               # 3.2 construct new blob for subsequent udl based on query_ids
               query_embeddings_for_cluster = query_embeddings[query_ids]
               print(f"keystring is {key_string}, query_embeddings_for_cluster shape: {query_embeddings_for_cluster.shape}")
               cascade_context.emit(key_string, query_embeddings_for_cluster,message_id=message_id)
               print(f"emitted!")
               

     def __del__(self):
          pass
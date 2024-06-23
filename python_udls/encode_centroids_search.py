#!/usr/bin/env python3
import cascade_context
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger
from derecho.cascade.udl import UserDefinedLogic
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
          cluster_key = f"/rag/emb/centroid_batch0"
          res = self.capi.get(cluster_key)
          if res:
               emb = res.get_result()['value']
               self.centroids_embeddings = np.frombuffer(emb, dtype=np.float32)
               print(f"embeddings: {self.centroids_embeddings} \n shape: {self.centroids_embeddings.shape}")
          self.have_centroids_loaded = True


     def ocdpo_handler(self,**kwargs):
          key = kwargs["key"]
          blob = kwargs["blob"]
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
          print(f"shape of query embeddings: {query_embeddings.shape}")
          # # 2. search the centroids
          # # TODO: implement this! Below is direct copy from faiss examples
          # d = 64                           # dimension
          # nb = 100000                      # database size
          # nq = 10000                       # nb of queries
          # np.random.seed(1234)             # make reproducible
          # xb = np.random.random((nb, d)).astype('float32')
          # xb[:, 0] += np.arange(nb) / 1000.
          # xq = np.random.random((nq, d)).astype('float32')
          # xq[:, 0] += np.arange(nq) / 1000.
          # k = 4                          # we want to see 4 nearest neighbors
          # D, I = index.search(xb[:5], k) # sanity check
          # print(I)
          # print(D)
          # D, I = index.search(xq, k)     # actual search
          # print(I[:5])                   # neighbors of the 5 first queries
          # print(I[-5:])                  # neighbors of the 5 last queries
          # # 3. trigger the subsequent UDL by evict the query to the top M shards according to affinity set sharding policy
          # # TODO: implement this

     def __del__(self):
          pass
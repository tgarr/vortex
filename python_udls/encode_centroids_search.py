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
          model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
     

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
          pass



     def ocdpo_handler(self,**kwargs):
          key = kwargs["key"]
          blob = kwargs["blob"]
          print("PYTHON Encode_centroids_search received key: ", key)
          array = np.array([1.1, 2.22, 3.333, 4.4444, 5.55555], dtype=np.float32)
          cascade_context.emit(key, array)
          # # 0. parse the query from blob
          # # TODO: below is a placehold implement this
          # query = [
          #      "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction."
          # ]
          # # 1. encode the query
          # query_embeddings = model.encode(
          #      query, return_dense=True, return_sparse=True, return_colbert_vecs=True
          # )
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
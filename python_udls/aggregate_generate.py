#!/usr/bin/env python3
import cascade_context
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger
from derecho.cascade.udl import UserDefinedLogic
from collections import defaultdict
import heapq
import io
import numpy as np
import json
import pickle
import re
import time
from perf_config import *
import transformers
import torch
from collections import OrderedDict


# Class to store the cluster search results for each query
class ClusterSearchResults:
     def __init__(self, cluster_counts, top_k, query_text):
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
          self.query_text = query_text
          
     def select_top_k(self):
          '''
          Select the top top_k results from all cluster results
          @return a list of top_k cluster_id and emb_id, ordered from closest to farthest
          '''
          # min heap to store the top_k results TODO: double check this
          all_results = []
          for cluster_id, embeddings in self.cluster_results.items():
               for emb_id, distance in embeddings.items():
                    if emb_id == "query":
                         self.query_text = distance
                    else:
                         all_results.append((float(distance), int(cluster_id), int(emb_id)))
          top_k_results = heapq.nsmallest(self.top_k, all_results, key=lambda x: x[0])
          return top_k_results


     def add_cluster_result(self, cluster_id, cluster_search_results):
          '''
          @param cluster_search_results: the search results for this cluster. 
                                        It is a dictionary in this format{emb_id:"distance"}
          '''
          self.cluster_results[cluster_id] = cluster_search_results
          if len(self.cluster_results) == self.cluster_counts:
               self.agg_top_k_res = self.select_top_k()
               self.collected_all_results = True
               

# TODO: this implementation has a lot of copies for small objects (query results), 
#       could be optimized if implemented in C++. But may not be the bottleneck
class AggregateGenerateUDL(UserDefinedLogic):
     def load_llm(self,):
          self.pipeline = transformers.pipeline(
               "text-generation",
               model=self.model_id,
               model_kwargs={"torch_dtype": torch.bfloat16},
               device_map="auto",
          )
          self.terminators = [
               self.pipeline.tokenizer.eos_token_id,
               self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
          ]
          

     def __init__(self,conf_str):
          '''
          Constructor
          '''
          # collect the cluster search result {(query_batch_key,query_count):{query_id: ClusterSearchResults, ...}, ...}
          self.cluster_search_res = {}
          # collect the LLM result per client_batch {(query_batch_key,query_count):{query_id: LLMResult, ...}, ...}
          self.llm_res = {}
          self.conf = json.loads(conf_str)
          self.top_k = int(self.conf["top_k"])
          self.top_clusters_count = int(self.conf["top_clusters_count"])
          self.capi = ServiceClientAPI()
          self.my_id = self.capi.get_my_id()
          self.tl = TimestampLogger()
          self.doc_file_name = './perf_data/miniset/doc_list.pickle'
          self.answer_mapping_file = './perf_data/miniset/answer_mapping.pickle'
          self.doc_list = None
          self.answer_mapping = None
          self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
          self.pipeline = None
          self.terminators = None
          # one server setting, gpu only host one model
          if not self.my_id == 0:
               self.load_llm()
          

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
          {
               qid: {
                    'doc_idx': [(distance, centroids_id, embedding_id), ...],
                    'query_text': query in natural language. 
                    }
               ...
          }
          '''
          client_query_batch_result = {}
          for qid in range(query_count):
               client_query_batch_result[qid] = \
                    {'doc_idx': self.cluster_search_res[(query_batch_key, query_count)][qid].agg_top_k_res,
                     'query_text': self.cluster_search_res[(query_batch_key, query_count)][qid].query_text
                    }
          return client_query_batch_result
     
     
     def _get_doc(self, cluster_id, ebd_id):
          """Helper method to get a piece of document in natural language.
          @input cluster_id: The id of the KNN cluster where the document falls in.
          @input ebd_id: The id of the document within the cluster.
          @return: The document string in natural language.
          """
          if self.answer_mapping is None:
               with open(self.answer_mapping_file, "rb") as file:
                    self.answer_mapping = pickle.load(file)
          if self.doc_list is None:
               with open(self.doc_file_name, 'rb') as file:
                    self.doc_list = pickle.load(file)
          return self.doc_list[self.answer_mapping[cluster_id][ebd_id]]
          

     def retrieve_documents(self, qury_result):
          """
          @input query_result
          {
               qid: {
                    'doc_idx': [(distance, centroids_id, embedding_id), ...],
                    'query_text': query in natural language. 
                    }    
               ...
          }
          @return all in natural language
          {
               query_text : [document_1, document_2, ...]
               ...
          }
          """     

          res = dict()
          for _, query_dict in qury_result.items():
               doc_list = [None] * len(query_dict['doc_idx'])
               query_text = query_dict['query_text']
               for idx, doc in enumerate(query_dict['doc_idx']):
                    doc_list[idx] = self._get_doc(doc[1], doc[2])
               res[query_text] = doc_list
          return res

     def llm_generate(self, sorted_client_query_batch_result):
          """
          @input all in natural language
          {
               query_text : [document_1, document_2, ...]
               ...
          }
          
          @return LLM gen answers
          {
               query_text : query_llm_response
          }
          
          
          for ordering of dictionary, the ref: https://stackoverflow.com/questions/1867861/how-to-keep-keys-values-in-same-order-as-declared
          """    
          batch_query_llm_response = OrderedDict()
          for query_text, doc_list in sorted_client_query_batch_result.items():
               messages = [
                    {"role": "system", "content": "Answer the user query based on this list of documents:"+" ".join(doc_list)},
                    {"role": "user", "content": query_text},
               ]
               
               tmp_res = self.pipeline(
                    messages,
                    max_new_tokens=256,
                    eos_token_id=self.terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
               )
               raw_text = tmp_res[0]["generated_text"][-1]['content']
               print(f"for query:{query_text}")
               print(f"the llm generated response: {raw_text}")
               batch_query_llm_response[query_text] = raw_text     
          return batch_query_llm_response
          
               
     
     
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

          qb_index = query_batch_key.find("qb")
          if PRINT_DEBUG_MESSAGE == 1:
               print(f"[AGGUDL]: query_batch_key: {query_batch_key}, \
                     query_batch_id:{query_batch_key[qb_index+2:]} \
                     qid: {qid}.")
          query_batch_id = int(query_batch_key[qb_index+2:]) #TODO: double check if there are other
          qb_qid = query_batch_id * 1000 * QUERY_PER_BATCH + qid 
          self.tl.log(LOG_TAG_AGG_UDL_START, self.my_id, qb_qid, cluster_id)
          

          # 1. parse the blob to dict
          bytes_obj = blob.tobytes()
          json_str_decoded = bytes_obj.decode('utf-8')
          cluster_result = json.loads(json_str_decoded)
          query_text = cluster_result["query"]
          
          # 2. add the cluster result to the aggregated query results
          if (query_batch_key, query_count) not in self.cluster_search_res:
               self.cluster_search_res[(query_batch_key, query_count)] = {}
          if qid not in self.cluster_search_res[(query_batch_key, query_count)]:
               self.cluster_search_res[(query_batch_key, query_count)][qid] =\
                    ClusterSearchResults(self.top_clusters_count, self.top_k, query_text)
          self.cluster_search_res[(query_batch_key,query_count)][qid].add_cluster_result(cluster_id, cluster_result)
          if not self.cluster_search_res[(query_batch_key,query_count)][qid].collected_all_results:
               self.tl.log(LOG_TAG_AGG_UDL_END, self.my_id, qb_qid, cluster_id)
               return

          self.tl.log(LOG_TAG_AGG_UDL_QUERY_FINISHED_GATHERED, self.my_id, qb_qid, 0)


          # If the code gets here, then it means that qid has collected all the results
          # TODO: get the documents, and run LLM for this query. Save the LLM result to self.llm_res

          # 3. check if all results for this batch of queries are collected
          if self.check_client_batch_finished(query_batch_key, query_count):
               # 3.1 format the key and value to be saved 
               next_key = f"/rag/generate/{query_batch_key}_results"
               client_query_batch_result = self.format_client_batch_result(query_batch_key, query_count)
               sorted_client_query_batch_result = {k: client_query_batch_result[k] for k in sorted(client_query_batch_result)}
               print(f"Yifan:\n {sorted_client_query_batch_result}")
               sorted_client_query_batch_result = self.retrieve_documents(sorted_client_query_batch_result)
               llm_generated_client_batch_res = self.llm_generate(sorted_client_query_batch_result)
               
               client_query_batch_result_json = json.dumps(llm_generated_client_batch_res)
               # 3.2 save the result as KV object in cascade to be retrieved by client
               self.tl.log(LOG_TAG_AGG_UDL_PUT_RESULT_START, self.my_id, query_batch_id, 0)
               self.capi.put(next_key, client_query_batch_result_json.encode('utf-8'))
               if PRINT_DEBUG_MESSAGE == 1:
                    print(f"[AggregateGenerate] put the agg_results to key:{next_key},\
                              \n                   value: {client_query_batch_result_json}")
               self.tl.log(LOG_TAG_AGG_UDL_PUT_RESULT_END, self.my_id, query_batch_id, 0)
          self.tl.log(LOG_TAG_AGG_UDL_END, self.my_id, qb_qid, cluster_id)
          

     def __del__(self):
          pass
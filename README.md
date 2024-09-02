# rag_demo

## Requirement
1. FAISS : https://github.com/facebookresearch/faiss/blob/main/INSTALL.md 

   NVIDIA toolkit nvcc if run with GPU support

   Python support needs to be built for Python centroids_search_udl
2. cascade : https://github.com/Derecho-Project/cascade/tree/7647a5f7c55aaed9327b62bc6bac93e1cbfb1223 , with Cascade Python API installed (https://github.com/Derecho-Project/cascade/tree/7647a5f7c55aaed9327b62bc6bac93e1cbfb1223/src/service/python )



# Configuration
The configuration of the udls could be found in dfgs.json file.


# Run

## Server Commands
In the main branch of this repo, we provide a bare-minimal setup of cascade server and client. It only requires two server nodes to start the service. n0, n1. run ``` ./run.sh server ``` under the build directory of the corresponding folder starts the service.


## Client Commands
In this repo of client and server configuration, we use node n2 as client node.

### 1. Object pools
The object pools needed for this pipeline: /rag/emb, /rag/doc, /rag/generate. The configurations of these object pools are in ```/setup/object_pool.list```. The first step of the client node is to create object pools needed by the pipeline.

### 2. put embeddings and documents
Construct the vector database by putting centroids and clusters' embeddings and documents.

#### Vector database Data Storage formats
- embeddings: stored in /rag/emb object pool

centroids stored in the format of /rag/emb/centroids_obj/[obj_id], e.g. /rag/emb/centroids_obj/1, /rag/emb/centroids_obj/2

cluster embeddings stored in the format of /rag/emb/cluster[cluster_id]/[obj_id], e.g. /rag/emb/cluster1/0, /rag/emb/cluster2/0

Note that because we use these keys as identifier to the embeddings object, if accidentally put other objects with the same prefix put to Cascade, it could cause unexpected knn search result. 

- documents: stored in cascade as KV objects under /rag/doc object pool in PCSS. Document objects' keys are in the format of [doc_path] = /rag/doc/[doc_identifier]

- embeddings to document path table. To fetch the document on a given embedding from its cluster_id and embedding_id. We keep a table for each cluster. The table matches the embeddings of that cluster to their corresponding pathnames. Using this table, the stored documents could be retrieved as context for the LLM. The tables are stored in Cascade in K/V format, with key as /rag/doc/emb_doc_map/cluster[cluster_id]/[table_id], value is in json with emb_id, document_pathname. There could be more than one table object per cluster, depends on the size of the cluster.



### 3. UDLs
After the vector database is constructed, clients could send batch of queries to cascade service. Queries are triggered by putting KV objects to its first UDL, encode_centroids_search_udl. 

- The key prefix to trigger this udl is /rag/emb/centroids_search/, which defined in /cfg/dfgs.json. After the key prefix, the key could have the identifier for this batch of requests as its suffix. The recommended format is "/rag/emb/centroids_search/client[client_id]_qb[query_batch_id]" (e.g. /rag/emb/centroids_search/client5_qb0). (query_batch_id is not required but used for logging purpose)

- The value is a json ({"query_text": ["query0", "query1"], "query_embedding": [emb0, emb1]}) in bytes formats.




### 4. Performance test
Client node is run in node n2. For performance testing, we created client side test scripts under the folder ```/perf_test```. 

#### Dataset
The dataset used for perf_test could be download from directory ```/setup/perf_data```. There are two datasets that prepared for experiments.

-  miniset is a subset of MSMARCO dataset(https://microsoft.github.io/msmarco/). It has around 800 documents, and used for a fast-check of the system. You can download it by running ```./download_miniset.sh```

- hotpot15 is from hotpot dataset(https://hotpotqa.github.io). It contains 899,667 documents with 11,5104,539 tokens, and used for performance testing. We have run encoding of these documents to embeddings using text-embedding-3-small from OpenAI API  and knn clustering them into 15 clusters using FAISS. You can download it by running ```./download_hotpot.sh```

#### Initialize database
The initialization step is to put the embeddings and documents to store in Cascade and use at query runtime. We provided a scrip that puts centroids' and clusters' embeddings, documents, and embedding-to-document-pathname map into Cascade. 

You can run ```python perf_test/perf_test_setup.py <dataset_directory>```. The ```<dataset_directory>``` could be either ```setup/perf_data/miniset``` or ```setup/perf_data/hotpot15```, depends on the scale of the experiment.

#### Run queries
After initialize the database, you can start to experiment with putting queries to Vortex and get the result. 
- latency experiment client. We wrote a program for testing latency of the pipeline. You can run via  ```./latency_client -n <num_requests> -b <batch_size> -q <dataset_director> -i <interval_between_request>```. 


# Docker image
We have built a docker image that have nvcc and cascade,derecho built and installed. You can pull from the docker image and run it on your environment.

Image name: yy354/rag_dev:v1.0

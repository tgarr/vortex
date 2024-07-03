# rag_demo

## Requirement
1. FAISS : https://github.com/facebookresearch/faiss/blob/main/INSTALL.md 

   NVIDIA toolkit nvcc if run with GPU support

   Python support needs to be built for Python centroids_search_udl
2. cascade : https://github.com/Derecho-Project/cascade/tree/7647a5f7c55aaed9327b62bc6bac93e1cbfb1223
3. cascade Python API : https://github.com/Derecho-Project/cascade/tree/7647a5f7c55aaed9327b62bc6bac93e1cbfb1223/src/service/python
4. model dependencies:

   BGE-M3: https://github.com/FlagOpen/FlagEmbedding 

   Generator model


# Run
## Server Commands
This is a bare-minimal setup of cascade server. It only requires two server nodes to start the service. n0, n1. run ``` ./run.sh server ``` under the build directory of the corresponding folder starts the service.

## Client Commands
### 1. create object pools
Initialize the service by creating object pools needed for this pipeline: /rag/emb, /rag/doc, /rag/generate. The configurations of these object pools are in ```/setup/object_pool.list```

### 2. put embeddings
Construct the vector database by putting centroids and clusters' embeddings and documents.

#### Vector database Data Storage formats
- embeddings: stored in /rag/emb object pool

centroids stored in the format of /rag/emb/centroids/[obj_id], e.g. /rag/emb/centroids/1, /rag/emb/centroids/2

cluster embeddings stored in the format of /rag/emb/cluster[cluster_id]/[obj_id], e.g. /rag/emb/cluster1/0, /rag/emb/cluster2/0

- documents: stored in cascade as KV objects under /rag/doc object pool. Document objects' keys are in the format of /rag/doc/[cluster_id]-[emb_id]

Step1 and step2 could be done by running ``` python setup.py ``` at client node, n4.

### 3. run UDLs
After the vector database is constructed, clients could send batch of queries to cascade service. Queries are triggered by put KV objects to its first UDL, encode_centroids_search_udl. The key prefix to trigger this udl is /rag/emb/py_centroids_search/, which defined in /cfg/dfgs.json.tmp. After the key prefix, it is required to have the client identifier and query_batch identifier attached to this key, the format is "/rag/emb/py_centroids_search/client[client_id]_qb[query_batch_id]" (e.g. /rag/emb/py_centroids_search/client5_qb0).

We wrote an example query using cascade python client API, client_query.py. One can test and run it in any client nodes, n4, n5 or n6.


# Docker image
We have built a docker image that have nvcc and cascade,derecho built setup. You can pull from the docker image and run it on your environment.

Image name: yy354/rag_dev:v1.0
# rag_demo

## Requirement
1. FAISS : https://github.com/facebookresearch/faiss/blob/main/INSTALL.md 

   NVIDIA toolkit nvcc if run with GPU support
2. cascade : https://github.com/Derecho-Project/cascade/tree/7647a5f7c55aaed9327b62bc6bac93e1cbfb1223
3. cascade Python API : https://github.com/Derecho-Project/cascade/tree/7647a5f7c55aaed9327b62bc6bac93e1cbfb1223/src/service/python

## Branches
We implemented two pipeline for RAG. 
- branch: cpp_udl. It has two C++ UDLs (centroids_search_udl, clusters_search_udl), and one Python udl (generate_udl)
- branch: py_cpp_udl. This one contains two Python UDL (centroids_search_udl, generate_udl) and one C++ UDL (clusters_sesarch_udl).

# Run
## Commands
1. create object pools
2. put embeddings
3. run UDLs


## Data Storage configuration
- embeddings: stored in /rag/emb object pool

centroids stored in the format of /rag/emb/centroid_[obj_id], e.g. /rag/emb/centroid_file1, /rag/emb/centroid_file2

cluster embeddings stored in the format of /rag/emb/clusters/cluster[cluster_id]_[obj_id], e.g. /rag/emb/clusters/cluster1_0, /rag/emb/clusters/cluster2_0


# Docker image
We have built a docker image that have nvcc and cascade,derecho built setup. You can pull from the docker image and run it on your environment.

Image name: yy354/rag_dev:v1.0
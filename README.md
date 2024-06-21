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


# Additional Note
python_udl.cpp and its corresponding CMake contents (in CMakeLists.txt, config.h.in, cascade_python_udlConfig.cmake.in) are copied from Weijia's code in Cascade repo: 
https://github.com/Derecho-Project/cascade/tree/master/src/udl_zoo/python

Copied them here to make the build process smooth, without the need to use pathname in udl_dll.cfg to direct to where udl_zoo/python built located.
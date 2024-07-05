#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_map>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>



class GroupedEmbeddings{
     int emb_dim;  //  e.g. 512. The dimension of each embedding
     int num_embs;  //  e.g. 1000. The number of embeddings in the array

     float* embeddings; 

     std::unique_ptr<faiss::IndexFlatL2> cpu_flatl2_index; // FAISS index object. Initialize if use CPU Flat search
     std::unique_ptr<faiss::gpu::StandardGpuResources> gpu_res;  // FAISS GPU resources. Initialize if use GPU search
     std::unique_ptr<faiss::gpu::GpuIndexFlatL2> gpu_flatl2_index; // FAISS index object. Initialize if use GPU Flat search
     std::unique_ptr<faiss::gpu::GpuIndexIVFFlat> gpu_ivf_flatl2_index; // FAISS index object. Initialize if use GPU IVF search

public:

     GroupedEmbeddings(int emb_dim, int num_embs, float* embeddings) 
          : emb_dim(emb_dim), num_embs(num_embs), embeddings(embeddings) {

     }

     /*** 
      * Initialize the CPU flat search index based on the embeddings.
      * initalize it if use faiss_cpu_flat_search()
     ***/
     void initialize_cpu_flat_search(){
          this->cpu_flatl2_index = std::make_unique<faiss::IndexFlatL2>(this->emb_dim); 
          this->cpu_flatl2_index->add(this->num_embs, this->embeddings); // add vectors to the index
     }

     /***
      * FAISS knn flat search on CPU
      * https://github.com/facebookresearch/faiss/blob/main/tutorial/cpp/1-Flat.cpp
      * @param nq: number of queries
      * @param xq: flaten queries to search 
      * @param top_k: number of top embeddings to return
      * @param D: distance array to store the distance of the top_k embeddings
      * @param I: index array to store the index of the top_k embeddings
     ***/
     int faiss_cpu_flat_search(int nq, float* xq, int top_k, float* D, long* I){
          if (PRINT_DEBUG_MESSAGE == 1)
               std::cout << "FAISS CPU flat Search in [GroupedEmbeddings] class" << std::endl;
          this->cpu_flatl2_index->search(nq, xq, top_k, D, I);
          return 0;
     }

     /*** 
      * Initialize the GPU flat search index based on the embeddings.
      * initalize it if use faiss_gpu_flat_search()
     ***/
     void initialize_gpu_flat_search(){
          this->gpu_res = std::make_unique<faiss::gpu::StandardGpuResources>();
          this->gpu_flatl2_index = std::make_unique<faiss::gpu::GpuIndexFlatL2>(this->gpu_res.get(), this->emb_dim);
          this->gpu_flatl2_index->add(this->num_embs, this->embeddings); // add vectors to the index
     }

     /***
      * FAISS knn flat l2 search on GPU
      * https://github.com/facebookresearch/faiss/blob/main/tutorial/cpp/4-GPU.cpp
      * @param nq: number of queries
      * @param xq: flaten queries to search 
      * @param top_k: number of top embeddings to return
      * @param D: distance array to store the distance of the top_k embeddings
      * @param I: index array to store the index of the top_k embeddings
     ***/
     int faiss_gpu_flat_search(int nq, float* xq, int top_k, float* D, long* I){
          if (PRINT_DEBUG_MESSAGE == 1)
               std::cout << "FAISS GPU flatl2 Search in [GroupedEmbeddings] class" << std::endl;
          int k = top_k;
          this->gpu_flatl2_index->search(nq, xq, k, D, I);
          return 0;
     }

     /*** 
      * Initialize the GPU ivf search index based on the embeddings.
      * initalize it if use faiss_gpu_ivf_flat_search()
     ***/
     void initialize_gpu_ivf_flat_search(){
          int nlist = 100;
          this->gpu_res = std::make_unique<faiss::gpu::StandardGpuResources>();
          this->gpu_ivf_flatl2_index = std::make_unique<faiss::gpu::GpuIndexIVFFlat>(this->gpu_res.get(), this->emb_dim, nlist, faiss::METRIC_L2);
          this->gpu_ivf_flatl2_index->add(this->num_embs, this->embeddings); // add vectors to the index
     }

     /***
      * FAISS knn search based on ivf search on GPU
      * https://github.com/facebookresearch/faiss/blob/main/tutorial/cpp/4-GPU.cpp
      * @param nq: number of queries
      * @param xq: flaten queries to search 
      * @param top_k: number of top embeddings to return
      * @param D: distance array to store the distance of the top_k embeddings
      * @param I: index array to store the index of the top_k embeddings
     ***/
     int faiss_gpu_ivf_flat_search(int nq, float* xq, int top_k, float* D, long* I){
          if (PRINT_DEBUG_MESSAGE == 1)
               std::cout << "FAISS GPU ivf flatl2 Search in [GroupedEmbeddings] class" << std::endl;
          this->gpu_ivf_flatl2_index->search(nq, xq, top_k, D, I);
          // print results
          printf("I (5 first results)=\n");
          for (int i = 0; i < 5; i++) {
               for (int j = 0; j < top_k; j++)
                    printf("%5ld ", I[i * top_k + j]);
               printf("\n");
          }

          printf("I (5 last results)=\n");
          for (int i = nq - 5; i < nq; i++) {
               for (int j = 0; j < top_k; j++)
                    printf("%5ld ", I[i * top_k + j]);
               printf("\n");
          }
          return 0;
     }

     ~GroupedEmbeddings() {
          // free(embeddings);
          delete[] this->embeddings;
     }

};
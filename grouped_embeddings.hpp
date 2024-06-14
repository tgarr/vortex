#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <unordered_map>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>



class GroupedEmbeddings{
     size_t unit_emb_size;  //  e.g. 512  Maybe should name it vector_size?
     
     size_t num_embeddings;  //  e.g. 1000 
     // 1D flattened array. e.g. size is 512 * 1000, float is 32-bit IEEE
     // Follow from FAISS xq, xb, which are both 1D flattend arrays 
     // maybe need to switch it to 2D array for dynamic RAG, to make insertion of new embeddings easier
     float* embeddings; 

     // the embeddings are concatenated and stored in KV object
     // given the object size limit, there might be multiple KV objects for embeddings of a group
     // allocation of embeddings object overprovision memory to avoid frequent reallocation
     size_t num_obj;  
     size_t emb_per_obj;  // e.g. 100

public:
     GroupedEmbeddings(size_t unit_emb_size, size_t num_obj, size_t emb_per_obj) {
          this->unit_emb_size = unit_emb_size;
          this->num_obj = num_obj;
          this->emb_per_obj = emb_per_obj;
          this->embeddings = (float*)malloc(unit_emb_size * num_embeddings * sizeof(float));
     }

     // Add new embeddings into the embeddings array
     void add_new_embeddings(float* embeddings, size_t num_add_embs) {
          if (this->num_embeddings + num_add_embs >= this->num_obj * this->emb_per_obj) {
               size_t new_num_ebds = this->num_embeddings + num_add_embs;
               this->num_obj = (new_num_ebds % this->emb_per_obj == 0) ? (new_num_ebds / this->emb_per_obj) : (new_num_ebds / this->emb_per_obj + 1);
               size_t new_size = this->num_obj * this->emb_per_obj * this->unit_emb_size;
               this->embeddings = (float*)realloc(this->embeddings, new_size * sizeof(float));
          } 
          memcpy(this->embeddings + start_idx * unit_emb_size, embeddings, num_add_embs * unit_emb_size * sizeof(float));
          this->num_embeddings += num_add_embs;
     }

     /***
      * Function copied from FAISS repository example: 
      * https://github.com/facebookresearch/faiss/blob/main/tutorial/cpp/4-GPU.cpp
      * TODO: update this to our needed implementation and data type
     ***/
     static int faiss_gpu_search(){
          int d = 64;      // dimension
          int nb = 100000; // database size
          int nq = 10000;  // nb of queries

          std::mt19937 rng;
          std::uniform_real_distribution<> distrib;

          /*** TODO: xb or xq is actuall this->embeddings ***/
          float* xb = new float[d * nb]; 
          float* xq = new float[d * nq];

          for (int i = 0; i < nb; i++) {
               for (int j = 0; j < d; j++)
                    xb[d * i + j] = distrib(rng);
               xb[d * i] += i / 1000.;
          }

          for (int i = 0; i < nq; i++) {
               for (int j = 0; j < d; j++)
                    xq[d * i + j] = distrib(rng);
               xq[d * i] += i / 1000.;
          }

          faiss::gpu::StandardGpuResources res;

          // Using a flat index

          faiss::gpu::GpuIndexFlatL2 index_flat(&res, d);

          printf("is_trained = %s\n", index_flat.is_trained ? "true" : "false");
          index_flat.add(nb, xb); // add vectors to the index
          printf("ntotal = %ld\n", index_flat.ntotal);

          int k = 4;

          { // search xq
               long* I = new long[k * nq];
               float* D = new float[k * nq];

               index_flat.search(nq, xq, k, D, I);

               // print results
               printf("I (5 first results)=\n");
               for (int i = 0; i < 5; i++) {
                    for (int j = 0; j < k; j++)
                         printf("%5ld ", I[i * k + j]);
                    printf("\n");
               }

               printf("I (5 last results)=\n");
               for (int i = nq - 5; i < nq; i++) {
                    for (int j = 0; j < k; j++)
                         printf("%5ld ", I[i * k + j]);
                    printf("\n");
               }

               delete[] I;
               delete[] D;
          }

          // Using an IVF index

          int nlist = 100;
          faiss::gpu::GpuIndexIVFFlat index_ivf(&res, d, nlist, faiss::METRIC_L2);

          assert(!index_ivf.is_trained);
          index_ivf.train(nb, xb);
          assert(index_ivf.is_trained);
          index_ivf.add(nb, xb); // add vectors to the index

          printf("is_trained = %s\n", index_ivf.is_trained ? "true" : "false");
          printf("ntotal = %ld\n", index_ivf.ntotal);

          { // search xq
               long* I = new long[k * nq];
               float* D = new float[k * nq];

               index_ivf.search(nq, xq, k, D, I);

               // print results
               printf("I (5 first results)=\n");
               for (int i = 0; i < 5; i++) {
                    for (int j = 0; j < k; j++)
                         printf("%5ld ", I[i * k + j]);
                    printf("\n");
               }

               printf("I (5 last results)=\n");
               for (int i = nq - 5; i < nq; i++) {
                    for (int j = 0; j < k; j++)
                         printf("%5ld ", I[i * k + j]);
                    printf("\n");
               }

               delete[] I;
               delete[] D;
          }

          delete[] xb;
          delete[] xq;

          return 0;

     }

     ~GroupedEmbeddings() {
          free(embeddings);
     }

};
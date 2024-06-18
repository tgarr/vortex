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
     int emb_dim;  //  e.g. 512. The dimension of each embedding
     
     int num_embs;  //  e.g. 1000. The number of embeddings in the array

     // 1D flattened array. e.g. size is 512 * 1000, float is 32-bit IEEE
     // Follow from FAISS xq, xb, which are both 1D flattend arrays 
     // maybe need to switch it to 2D array for dynamic RAG, to make insertion of new embeddings easier
     float* embeddings; 

     // the embeddings are concatenated and stored in KV object
     // given the object size limit, there might be multiple KV objects for embeddings of a group
     // allocation of embeddings object overprovision memory to avoid frequent reallocation
     int num_obj;  
     int emb_per_obj;  // e.g. 100

public:
     GroupedEmbeddings(int emb_dim, int num_obj, int emb_per_obj) {
          this->emb_dim = emb_dim;
          this->num_obj = num_obj;
          this->emb_per_obj = emb_per_obj;
          this->embeddings = (float*)malloc(emb_dim * num_embs * sizeof(float));
     }

     GroupedEmbeddings(int emb_dim, int num_embs, float* embeddings, int num_obj, int emb_per_obj) 
          : emb_dim(emb_dim), num_embs(num_embs), embeddings(embeddings), num_obj(num_obj), emb_per_obj(emb_per_obj) {
     }

     // Add new embeddings into the embeddings array
     void add_new_embeddings(float* embeddings, int num_add_embs) {
          if (this->num_embs + num_add_embs >= this->num_obj * this->emb_per_obj) {
               int new_num_embs = this->num_embs + num_add_embs;
               this->num_obj = (new_num_embs % this->emb_per_obj == 0) ? (new_num_embs / this->emb_per_obj) : (new_num_embs / this->emb_per_obj + 1);
               int new_size = this->num_obj * this->emb_per_obj * this->emb_dim;
               this->embeddings = (float*)realloc(this->embeddings, new_size * sizeof(float));
          } 
          memcpy(this->embeddings + this->num_embs * emb_dim, embeddings, num_add_embs * emb_dim * sizeof(float));
          this->num_embs += num_add_embs;
     }

     /***
      * Function copied from FAISS repository example: 
      * https://github.com/facebookresearch/faiss/blob/main/tutorial/cpp/4-GPU.cpp
      * TODO: update this to our needed implementation and data type
      * @param nq: number of queries
      * @param xq: flaten queries to search 
     ***/
     int faiss_gpu_search(int nq, float* xq){

          std::cout << "FAISS GPU Search in [GroupedEmbeddings] class" << std::endl;
          // int d = 64;      // dimension
          // int nb = 100000; // database size
          // int nq = 10000;  // nb of queries
          int d = this->emb_dim; // dimension
          int nb = this->num_embs; 

          std::mt19937 rng;
          std::uniform_real_distribution<> distrib;

          
          // float* xb = new float[d * nb]; 
          // float* xq = new float[d * nq];
          float* xb = this->embeddings;

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

          // printf("is_trained = %s\n", index_flat.is_trained ? "true" : "false");
          index_flat.add(nb, xb); // add vectors to the index
          // printf("ntotal = %ld\n", index_flat.ntotal);

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

          // delete[] xb;
          // delete[] xq;

          return 0;

     }

     /***
      * Function copied from FAISS repository example: 
      * https://github.com/facebookresearch/faiss/blob/main/tutorial/cpp/1-Flat.cpp
      * TODO: update this to our needed implementation and data type
      * @param nq: number of queries
      * @param xq: flaten queries to search 
     ***/
     int faiss_cpu_flat_search(int nq, float* xq){

          std::cout << "FAISS CPU flat Search in [GroupedEmbeddings] class" << std::endl;
          // int d = 64;      // dimension
          // int nb = 100000; // database size
          // int nq = 10000;  // nb of queries
          int d = this->emb_dim; // dimension
          int nb = this->num_embs; 

          std::mt19937 rng;
          std::uniform_real_distribution<> distrib;

          // float* xb = new float[d * nb];
          float* xb = this->embeddings;

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

          faiss::IndexFlatL2 index(d); // call constructor
          // printf("is_trained = %s\n", index.is_trained ? "true" : "false");
          index.add(nb, xb); // add vectors to the index
          // printf("ntotal = %zd\n", index.ntotal);

          int k = 4;

          { // sanity check: search 5 first vectors of xb
               faiss::idx_t* I = new faiss::idx_t[k * 5];
               float* D = new float[k * 5];

               index.search(5, xb, k, D, I);

               // print results
               printf("I=\n");
               for (int i = 0; i < 5; i++) {
                    for (int j = 0; j < k; j++)
                         printf("%5zd ", I[i * k + j]);
                    printf("\n");
               }

               printf("D=\n");
               for (int i = 0; i < 5; i++) {
                    for (int j = 0; j < k; j++)
                         printf("%7g ", D[i * k + j]);
                    printf("\n");
               }

               delete[] I;
               delete[] D;
          }

          { // search xq
               faiss::idx_t* I = new faiss::idx_t[k * nq];
               float* D = new float[k * nq];

               index.search(nq, xq, k, D, I);

               // print results
               printf("I (5 first results)=\n");
               for (int i = 0; i < 5; i++) {
                    for (int j = 0; j < k; j++)
                         printf("%5zd ", I[i * k + j]);
                    printf("\n");
               }

               printf("D (5 last results)=\n");
               for (int i = nq - 5; i < nq; i++) {
                    for (int j = 0; j < k; j++)
                         printf("%5f ", D[i * k + j]);
                    printf("\n");
               }

               delete[] I;
               delete[] D;
          }

          // delete[] xb;
          // delete[] xq;

          return 0;
     }

     ~GroupedEmbeddings() {
          free(embeddings);
     }

};
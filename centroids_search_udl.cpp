#include <cascade/user_defined_logic_interface.hpp>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

namespace derecho{
namespace cascade{

#define MY_UUID     "11a2c123-1100-21ac-1755-0001ac110000"
#define MY_DESC     "UDL search which centroids the queries close to."

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

class CentroidsSearchOCDPO: public OffCriticalDataPathObserver {

    /***
    * Function copied from FAISS repository example: 
    * https://github.com/facebookresearch/faiss/blob/main/tutorial/cpp/4-GPU.cpp
    ***/
    static int faiss_gpu_search(){
        int d = 64;      // dimension
        int nb = 100000; // database size
        int nq = 10000;  // nb of queries

        std::mt19937 rng;
        std::uniform_real_distribution<> distrib;

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

    virtual void operator () (const derecho::node_id_t sender,
                              const std::string& key_string,
                              const uint32_t prefix_length,
                              persistent::version_t version,
                              const mutils::ByteRepresentable* const value_ptr,
                              const std::unordered_map<std::string,bool>& outputs,
                              ICascadeContext* ctxt,
                              uint32_t worker_id) override {
        std::cout << "[centroids search ocdpo]: I(" << worker_id << ") received an object from sender:" << sender << " with key=" << key_string 
                  << ", matching prefix=" << key_string.substr(0,prefix_length) << std::endl;
        faiss_gpu_search();
    }

    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;
public:
    static void initialize() {
        if(!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<CentroidsSearchOCDPO>();
        }
    }
    static auto get() {
        return ocdpo_ptr;
    }
};

std::shared_ptr<OffCriticalDataPathObserver> CentroidsSearchOCDPO::ocdpo_ptr;

void initialize(ICascadeContext* ctxt) {
    CentroidsSearchOCDPO::initialize();
}

std::shared_ptr<OffCriticalDataPathObserver> get_observer(
        ICascadeContext*,const nlohmann::json&) {
    return CentroidsSearchOCDPO::get();
}

void release(ICascadeContext* ctxt) {
    // nothing to release
    return;
}

} // namespace cascade
} // namespace derecho

#include <cascade/user_defined_logic_interface.hpp>
#include <iostream>
#include <unordered_map>
#include <memory>

#include "grouped_embeddings.hpp"


namespace derecho{
namespace cascade{

#define MY_UUID     "11a2c123-2200-21ac-1755-0002ac220000"
#define MY_DESC     "UDL search within the clusters to find the top K embeddings that the queries close to."

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

class ClustersSearchOCDPO: public DefaultOffCriticalDataPathObserver {

    // maps from cluster ID -> embeddings of that cluster, 
    // because there could be more than 1 clusters hashed to one node by affinity set.
    std::unordered_map<uint32_t, std::unique_ptr<GroupedEmbeddings>> clusters_embs;
    bool is_clusters_embs_cached = false;
    /*** TODO: get this from dfgs config ***/
    // faiss example uses 64
    int emb_dim = 64; // dimension of each embedding
    /*** TODO: this is a duplicated field to num_embs of GroupedEmbeddings ***/
    // same as faiss example
    int num_embs = 100000; // number of embeddings


    /***
     * Fill in the memory cache by getting the clusters' embeddings from KV store in Cascade
     * This function is called when this UDL is first triggered by caller to operator(),
     * it sacrifices the first request to this node, but the following requests will benefit from this cache.
     * In static RAG setting, this function should be called only once at the begining
     * In dynamic RAG setting, this function could be extended to call periodically or upon notification 
     * (The reason of not filling it at initialization, is that initialization is called upon server starts, 
     *  but the data have not been put to the servers yet, this needs to be called after the clusters' embeddings data are put)
    ***/
    void fill_in_cached_clusters_embs(){
        /*** TODO: implement this! ***/
        // 1. get the cluster embeddings from KV store in Cascade
        float* xb = new float[this->emb_dim * this->num_embs]; // Placeholder embeddings
        // 2. fill in the memory cache
        this->clusters_embs[0]= std::make_unique<GroupedEmbeddings>(this->emb_dim, this->num_embs, xb, 1, 10000);
        // 3. set the flag to true
        is_clusters_embs_cached = true;
        return;
    }

    virtual void ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) override {
        std::cout << "[Clusters search ocdpo]: I(" << worker_id << ") received an object from sender:" << sender << " with key=" << key_string 
                  << ", object_pool_pathname=" << object_pool_pathname << std::endl;
        // 0. check if local centroids cache is filled
        if (!is_clusters_embs_cached) {
            fill_in_cached_clusters_embs();
        }
        // 1. compute knn for the corresponding cluster on this node
        // 1.0. get the cluster ID
        uint32_t cluster_id = 0; // TODO: get the cluster ID from the object
        // 1.1. get the embeddings of the cluster
        auto& embs = clusters_embs[cluster_id];
        // 1.2. search the top K embeddings that are close to the query
        int nq = 10000;
        float* xq = new float[this->emb_dim * nq]; // Placeholder query embeddings
        embs->faiss_gpu_search(nq, xq);
        // 1.3. send the top K embeddings to the next stage
        
        // 2. emit the result to the subsequent UDL
        // Blob blob(object.blob);
        // emit(key_string, EMIT_NO_VERSION_AND_TIMESTAMP , blob);
    }

    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;
public:

    static void initialize() {
        if(!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<ClustersSearchOCDPO>();
        }
    }
    static auto get() {
        return ocdpo_ptr;
    }
};

std::shared_ptr<OffCriticalDataPathObserver> ClustersSearchOCDPO::ocdpo_ptr;

void initialize(ICascadeContext* ctxt) {
    ClustersSearchOCDPO::initialize();
}

std::shared_ptr<OffCriticalDataPathObserver> get_observer(
        ICascadeContext*,const nlohmann::json&) {
    return ClustersSearchOCDPO::get();
}

void release(ICascadeContext* ctxt) {
    // nothing to release
    return;
}

} // namespace cascade
} // namespace derecho

#include <condition_variable>
#include <memory>
#include <map>
#include <iostream>
#include <shared_mutex>
#include <unordered_map>

#include <cascade/user_defined_logic_interface.hpp>
#include <cascade/utils.hpp>
#include <cascade/cascade_interface.hpp>
#include <derecho/openssl/hash.hpp>

#include "grouped_embeddings_for_search.hpp"
#include "api_utils.hpp"


namespace derecho{
namespace cascade{

#define MY_UUID     "11a2c123-2200-21ac-1755-0002ac220000"
#define MY_DESC     "UDL search within the clusters to find the top K embeddings that the queries close to."

#define CLUSTER_EMB_OBJECTPOOL_PREFIX "/rag/emb/clusters/cluster"
#define EMIT_AGGREGATE_PREFIX "/rag/generate/agg"


std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}




class ClustersSearchOCDPO: public DefaultOffCriticalDataPathObserver {
    

    /***
     * A thread class to process ANN search on the queued queries in round-robin order on the local clusters
     *   then emit the result to the next UDL
     */
    class ClusterSearchWorker {
        uint64_t my_thread_id;
        ClustersSearchOCDPO* parent;

        bool running = false;
        std::thread real_thread;

        std::unique_ptr<queryQueue> query_buffer;
        std::unique_ptr<queryQueue> shadow_query_buffer;
        int use_shadow_flag;  // if this is 1, then add to shadow_query_buffer, otherwise add to query_buffer
        std::condition_variable_any query_buffer_cv;
        std::mutex query_buffer_mutex;
        

        /***
        * Format the new_keys for the search results of the queries
        * it is formated as client{client_id}qb{querybatch_id}qc{client_batch_query_count}_cluster{cluster_id}_qid{hash(query)}
        * @param new_keys the new keys to be constructed
        * @param key_string the key string of the object
        * @param query_list a list of query strings
        ***/
        void construct_new_keys(std::vector<std::string>& new_keys,
                                std::vector<std::string>::const_iterator keys_begin,
                                std::vector<std::string>::const_iterator keys_end,
                                std::vector<std::string>::const_iterator queries_begin,
                                std::vector<std::string>::const_iterator queries_end);

        // Helper function: Emit results to the next UDL
        void emit_results(DefaultCascadeContextType* typed_ctxt,
                        const std::vector<std::string>& new_keys,
                        const std::vector<std::string>& query_list,
                        long* I, float* D, size_t start_idx, size_t batch_size);
        /***
         * Run ANN algorithm on batch of queries from query_buffer and emit the results once the whole batch finishes
         *  used for batchable search, which is more performant when the number of queries is large
        */
        void run_cluster_search_and_emit(DefaultCascadeContextType* typed_ctxt,
                                        queryQueue* query_buffer);
        
        /*** Helper function to check if there are enough pending queries on
         *   the buffer that have been added by the push_thread
         *   If so, then flip the flag to have the push_thread to start a new buffer, 
         *   while this thread process the queued queries
         *   @param num the number of queries to check if enough pending
         *   @return true if the accumulated queries > min_batch_size
         */
        bool enough_pending_queries(int num);

    public:
        ClusterSearchWorker(uint64_t thread_id, 
                            ClustersSearchOCDPO* parent_udl);
        void main_loop(DefaultCascadeContextType* typed_ctxt);   
        void push_to_query_buffer(int cluster_id, const Blob& blob, const std::string& key);
        // void push_queries_to_cluster(int cluster_id, std::vector<std::string>& query_list, std::vector<std::string>& query_keys);     
        void start(DefaultCascadeContextType* typed_ctxt);
        void join();
        void signal_stop();
    };

    // These two values could be set by config in dfgs.json.tmp file
    int emb_dim = 64; // dimension of each embedding
    uint32_t top_k = 4; // number of top K embeddings to search
    int faiss_search_type = 0; // 0: CPU flat search, 1: GPU flat search, 2: GPU IVF search
    int my_id; // the node id of this node; logging purpose
    uint32_t min_batch_size = 1; // min number of queries to send down the pipeline
    uint32_t max_batch_size = 100; // max number of queries to send down the pipeline
    uint32_t batch_time_us = 1000; // the time to wait for the minimum batch size
    uint32_t min_process_batch_size = 1; // min number of queries to process in each batch
    uint32_t max_process_batch_size = 10; // max number of queries to process in each batch
                                          // for hnsw search, set it to a small value (e.g. 1 to 10): the goal is to better distribute the queries across threads
                                          // for GPU-based search, set it to a big value (e.g. 100s to 1000s): the goal is to better utilize GPU memory and cores
    uint32_t process_batch_time_us = 1000; // the time to wait for the minimum process batch

    int num_threads = 1; // number of threads to process the cluster search
    uint64_t next_thread = 0;

    int cluster_id = -1; // the cluster that this node handles
    mutable std::shared_mutex cluster_search_index_mutex;
    mutable std::condition_variable_any cluster_search_index_cv;
    // maps from cluster ID -> embeddings of that cluster, 
    // use pointer to allow multithreading adding queries to different GroupedEmbeddingsForSearch objects
    std::shared_ptr<GroupedEmbeddingsForSearch> cluster_search_index;    

    bool check_and_retrieve_cluster_index(DefaultCascadeContextType* typed_ctxt);
    virtual void ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) override;

    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;

public:
    /*** TODO: have a thread_pool for process cluster search workers */
    std::vector<std::unique_ptr<ClusterSearchWorker>> cluster_search_threads;

    static void initialize() {
        if(!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<ClustersSearchOCDPO>();
        }
    }
    static auto get() {
        return ocdpo_ptr;
    }

    void start_threads(DefaultCascadeContextType* typed_ctxt);

    void set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config);

    void shutdown();
};

std::shared_ptr<OffCriticalDataPathObserver> ClustersSearchOCDPO::ocdpo_ptr;

void initialize(ICascadeContext* ctxt) {
    ClustersSearchOCDPO::initialize();
}

std::shared_ptr<OffCriticalDataPathObserver> get_observer(
        ICascadeContext* ctxt,const nlohmann::json& config) {
    auto typed_ctxt = dynamic_cast<DefaultCascadeContextType*>(ctxt);
    std::static_pointer_cast<ClustersSearchOCDPO>(ClustersSearchOCDPO::get())->set_config(typed_ctxt,config);
    return ClustersSearchOCDPO::get();
}

void release(ICascadeContext* ctxt) {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::cerr << "Error during cudaDeviceSynchronize in release: " << cudaGetErrorString(sync_err) << std::endl;
    } 
    std::static_pointer_cast<ClustersSearchOCDPO>(ClustersSearchOCDPO::get())->shutdown();  
    return;
}

} // namespace cascade
} // namespace derecho

#include <memory>
#include <map>
#include <iostream>
#include <unordered_map>

#include "grouped_embeddings_for_search.hpp"
#include "api_utils.hpp"


namespace derecho{
namespace cascade{

#define MY_UUID     "10a2c111-1100-1100-1000-0001ac110000"
#define MY_DESC     "UDL search among the centroids to find the top num_centroids that the queries close to."
#define NEXT_UDL_SUBGROUP_ID 1 // cluster_search udl subgroup_id

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

struct batchedTask {
    std::string key;
    uint32_t client_id;
    uint32_t query_batch_id;
    Blob blob;

    // std::vector<std::string> queries;
    // // CURL *curl;
    // std::unique_ptr<float[]> all_embeddings;
    batchedTask(std::string key, uint32_t client_id, uint32_t query_batch_id, Blob blob)
        : key(key), client_id(client_id), query_batch_id(query_batch_id), blob(blob) {}
};

class CentroidsSearchOCDPO: public DefaultOffCriticalDataPathObserver {

    /***
     * A thread class to compute topk centroids and emit to the next UDL
     */
    class ProcessBatchedTasksThread {
        private:
            uint64_t my_thread_id;
            CentroidsSearchOCDPO* parent;
            std::thread real_thread;
            
            bool running = false;

            bool get_queries_and_emebddings(Blob* blob, 
                                    float*& data, uint32_t& nq, 
                                    std::vector<std::string>& query_list,
                                    const uint32_t& client_id, 
                                    const uint32_t& query_batch_id);
            /***
            * Combine subsets of queries that is going to send to the same cluster
            *  A batching step that batches the results with the same cluster in their top_num_centroids search results
            * @param I the indices of the top_num_centroids that are close to the queries
            * @param nq the number of queries
            * @param cluster_ids_to_query_ids a map from cluster_id to the list of query_ids that are close to the cluster
            ***/
            void combine_common_clusters(const long* I, const int nq, 
                                    std::map<long, std::vector<int>>& cluster_ids_to_query_ids);
            void process_task(std::unique_ptr<batchedTask> task_ptr, DefaultCascadeContextType* typed_ctxt);
            void main_loop(DefaultCascadeContextType* typed_ctxt);

        public:
            ProcessBatchedTasksThread(uint64_t thread_id, CentroidsSearchOCDPO* parent_udl);
            void start(DefaultCascadeContextType* typed_ctxt);
            void join();
            void signal_stop();
    };

    std::unique_ptr<GroupedEmbeddingsForSearch> centroids_embs;
    bool cached_centroids_embs = false;
    int my_id = -1; // id of this node; logging purpose

    // values set by config in dfgs.json.tmp file
    std::string centroids_emb_prefix = "/rag/emb/centroids_obj";
    int emb_dim = 64; // dimension of each embedding
    int top_num_centroids = 4; // number of top K embeddings to search
    int faiss_search_type = 0; // 0: CPU flat search, 1: GPU flat search, 2: GPU IVF search
    std::string emit_key_prefix = "/rag/emb/clusters_search";
    
    std::atomic<bool> new_request = false;
    std::mutex active_tasks_mutex;
    std::condition_variable active_tasks_cv;
    std::queue<std::unique_ptr<batchedTask>> active_tasks_queue;

    
    bool retrieve_and_cache_centroids_index(DefaultCascadeContextType* typed_ctxt);


    virtual void ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) override;

    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;


public:
    std::unique_ptr<ProcessBatchedTasksThread> process_batched_tasks_thread;
    static void initialize() {
        if(!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<CentroidsSearchOCDPO>();
        }
    }
    static auto get() {
        return ocdpo_ptr;
    }

    void set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config);

    void shutdown();
};

std::shared_ptr<OffCriticalDataPathObserver> CentroidsSearchOCDPO::ocdpo_ptr;

void initialize(ICascadeContext* ctxt) {
    CentroidsSearchOCDPO::initialize();
}

std::shared_ptr<OffCriticalDataPathObserver> get_observer(
        ICascadeContext* ctxt,const nlohmann::json& config) {
    auto typed_ctxt = dynamic_cast<DefaultCascadeContextType*>(ctxt);
    std::static_pointer_cast<CentroidsSearchOCDPO>(CentroidsSearchOCDPO::get())->set_config(typed_ctxt,config);
    return CentroidsSearchOCDPO::get();
}

void release(ICascadeContext* ctxt) {
    std::static_pointer_cast<CentroidsSearchOCDPO>(CentroidsSearchOCDPO::get())->shutdown();
    return;
}

} // namespace cascade
} // namespace derecho

#pragma once

#include <algorithm>
#include <memory>
#include <map>
#include <iostream>
#include <tuple>
#include <unordered_map>

#include <cascade/user_defined_logic_interface.hpp>
#include <cascade/utils.hpp>
#include <cascade/cascade_interface.hpp>

#include "serialize_utils.hpp"
#include "api_utils.hpp"

namespace derecho {
namespace cascade {

#define MY_UUID "11a3c123-3300-31ac-1866-0003ac330000"
#define MY_DESC "UDL to aggregate the knn search results for each query from the clusters and run LLM with the query and its top_k closest docs."

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

// map of client_id -> [query_texts, query_batch_id, qid]
using client_queries_map_t = std::unordered_map<int, std::vector<std::tuple<std::string, int, int>>>;

struct QuerySearchResults {
    const std::string query_text;
    int total_cluster_num = 0;
    std::vector<int> collected_cluster_ids;
    bool collected_all_results = false;
    int top_k = 0;
    std::priority_queue<DocIndex> agg_top_k_results;
    std::vector<std::string> top_k_docs;
    bool retrieved_top_k_docs = false;
    std::string api_result;

    QuerySearchResults(const std::string& query_text, int total_cluster_num, int top_k);
    bool is_all_results_collected();
    void add_cluster_result(int cluster_id, std::vector<DocIndex> cluster_results);
};

/*** Struct to keep track of if the query has been processed
 *   for repeated query text, use cached results that computed by previous queries
 */
struct QueryRequestSource {
    uint32_t client_id;
    uint32_t query_batch_id;
    uint32_t qid;
    int total_cluster_num;
    int received_cluster_result_count;
    bool notified_client;

    QueryRequestSource(uint32_t client_id, uint32_t query_batch_id,uint32_t qid ,int total_cluster_num, int received_cluster_result_count, bool notified_client);
};

struct queuedTask {
    int client_id;
    int batch_id;
    int qid;
    int query_batch_id;
    std::string query_text;
    int cluster_id;
    std::vector<DocIndex> cluster_results;
    queuedTask(): client_id(-1), batch_id(-1), qid(-1), query_batch_id(-1), query_text(""), cluster_id(-1), cluster_results() {}
};

class AggGenOCDPO : public DefaultOffCriticalDataPathObserver {

private:
    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;

    class ProcessThread {
        private:
            uint64_t my_thread_id;
            AggGenOCDPO* parent;
            std::thread real_thread;
            
            bool running = false;
            std::mutex thread_mtx;
            std::condition_variable thread_signal;

            std::queue<std::unique_ptr<queuedTask>> pending_tasks;

            /*** query_result: query_text -> QuerySearchResults 
             *   is a UDL local cache to store the cluster search results for queries that haven't notified the client 
             *   (due to not all cluster results are collected)
            */
            std::unordered_map<std::string, std::unique_ptr<QuerySearchResults>> query_results;
            std::unordered_map<std::string, std::future<std::string>> query_api_futures;
            /*** Since same query may appear in different query batches from different clients. i.e. different people ask the same question
             *   query_request_tracker: query_text -> [(client_id, query_batch_id, notified_client), ..]
             *   query_request_tracker keep track of the batched query requests that requested the same type of query.
             *   This is used as a helper field for caching the query_results, and early reply to the client if the results are ready
             *   to delay garbage collection of the results for a query if there is still requesting qb.
            */
            std::unordered_map<std::string, std::vector<QueryRequestSource>> query_request_tracker;

            
            /*** Helper function to check if the query result has been notified to the client
             *   for repeated query text
            */
            bool has_notified_client(const std::string& query_text, 
                                                const uint32_t& client_id, 
                                                const uint32_t& query_batch_id, 
                                                const uint32_t& qid);

            void serialize_results_and_notify_client(DefaultCascadeContextType* typed_ctxt, 
                                                    const std::vector<std::tuple<std::string, int, int>>& query_infos, 
                                                    int client_id);

            void garbage_collect_query_results(const std::string& query_text, 
                                            const uint32_t& client_id, const uint32_t& query_batch_id, 
                                            const uint32_t& qid);

            // Helper function to run LLM with top_k_docs asynchronously
            void async_run_llm_with_top_k_docs(const std::string& query_text);

            /***  Helper function to get the top_k docs for a query
             *    Based on the top_k embedding index of the cluster results, 
             *    get the corresponding emebdding global index and the original doc contents
             */
            bool get_topk_docs(DefaultCascadeContextType* typed_ctxt, std::string& query_text);

            /***
             *  Process individual task on the pending queue
             *  1. check if the query has been processed before
             *  Also handle the case where multiple different client send the same query 
             *  Because UDL2 computes ANN for all received queries without awaring of if the query has been processed,
             *  it triggers this UDL multiple time even after we send back the query result to the client already. 
             *  At aggregation step, we use the local cache to track whether the results have been collected for that query before,
             *  avoiding it to send the same result to the same client multiple times.
             *  2. add the cluster_results to the query_results and check if all results are collected
             *  @return true if the query_result is ready to be notified to the clients,
                    false otherwise
             */
            bool process_task(DefaultCascadeContextType* typed_ctxt, queuedTask* task_ptr);

            /***
             * Combine the queries to be replied to the same clients
             *  handle the case for repeating queries from different clients   
             * @param query_text: the query text
             * @param notify_info: the map of client_id -> [query_texts, query_batch_id, qid]
             */
            void get_clients_with_same_query(const std::string& query_text,client_queries_map_t& notify_info);

            /***
             *  process tasks on the pending queue
             *  @param typed_ctxt: the typed context of the cascade service
             *  @param tasks: a vector of tasks to process
             *  @param finished_queries: a vector to store the queries that have received all of its clusters' results 
             *                           and ready to be notified to client
             */
            void process_tasks(DefaultCascadeContextType* typed_ctxt, 
                                std::vector<std::unique_ptr<queuedTask>>& tasks,
                                std::vector<std::string>& finished_queries);

            /*** check llm async result from curl, 
             *   process the queries that have received llm generator results
             */
            void process_llm_async_results(std::vector<std::string>& finished_queries);
            
            void main_loop(DefaultCascadeContextType* typed_ctxt);

        public:
            ProcessThread(uint64_t thread_id, AggGenOCDPO* parent_udl);
            void start(DefaultCascadeContextType* typed_ctxt);
            void join();
            void signal_stop();
            void push_task(std::unique_ptr<queuedTask> task);
    };

    int top_k = 5; // final top K results to use for LLM
    int top_num_centroids = 4; // number of top K clusters need to wait to gather for each query
    int include_llm = false; // 0: not include, 1: include
    int retrieve_docs = true; // 0: not retrieve, 1: retrieve
    std::string openai_api_key;
    std::string llm_model_name = "gpt4o-mini";
    int min_batch_size = 1;
    int max_batch_size = 10;
    int batch_time_us = 100;
    int my_id;


    std::shared_mutex doc_cache_mutex;
    std::condition_variable doc_cache_cv;
     /*** TODO: could use a more efficient way to store the doc_contents cache */
    std::unordered_map<int, std::unordered_map<long, std::string>> doc_tables; // cluster_id -> emb_index -> pathname
    std::unordered_map<int, std::unordered_map<long, std::string>> doc_contents; // {cluster_id0:{ emb_index0: doc content0, ...}, cluster_id1:{...}, ...}
    

    /*** Helper function to load the doc table for a given cluster_id
     *   Use the doc_table could find the pathname of a text document that corresponds to a given cluster's embedding index
     */
    bool load_doc_table(DefaultCascadeContextType* typed_ctxt, int cluster_id);
    /*** Helper function to get a doc for a given cluster_id and emb_index. Used by get_topk_docs
     *   First check if the doc is in the cache, if not, cache it after retrieving it from cascade
     */
    bool get_doc(DefaultCascadeContextType* typed_ctxt, int cluster_id, long emb_index, std::string& res_doc);
    
    

    virtual void ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) override;

public:
    std::unique_ptr<ProcessThread> process_thread;
    static void initialize() {
        if(!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<AggGenOCDPO>();
        }
    }
    static auto get() {
        return ocdpo_ptr;
    }
    void start_threads(DefaultCascadeContextType* typed_ctxt);
    void set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config);
    void shutdown() ;
};

std::shared_ptr<OffCriticalDataPathObserver> AggGenOCDPO::ocdpo_ptr;

void initialize(ICascadeContext* ctxt) {
    AggGenOCDPO::initialize();
}


// std::shared_ptr<OffCriticalDataPathObserver> AggGenOCDPO::get() {
//     return ocdpo_ptr;
// }


std::shared_ptr<OffCriticalDataPathObserver> get_observer(ICascadeContext* ctxt, 
                                                        const nlohmann::json& config) {
    auto typed_ctxt = dynamic_cast<DefaultCascadeContextType*>(ctxt);
    std::static_pointer_cast<AggGenOCDPO>(AggGenOCDPO::get())->set_config(typed_ctxt, config);
    return AggGenOCDPO::get();
}

void release(ICascadeContext* ctxt) {
    std::static_pointer_cast<AggGenOCDPO>(AggGenOCDPO::get())->shutdown();
    return;
}

}  // namespace cascade
}  // namespace derecho


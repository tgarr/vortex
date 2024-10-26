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

std::string get_uuid();
std::string get_description();

// struct DocIndex; // declared in serialize_utils.hpp

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

struct QueryRequestSource {
    uint32_t client_id;
    uint32_t query_batch_id;
    int total_cluster_num;
    int received_cluster_result_count;
    bool finished_process;

    QueryRequestSource(uint32_t client_id, uint32_t query_batch_id, int total_cluster_num, int received_cluster_result_count, bool finished_process);
};

class AggGenOCDPO : public DefaultOffCriticalDataPathObserver {
public:
    static void initialize();
    static std::shared_ptr<OffCriticalDataPathObserver> get();
    void set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config);

private:
    int top_k = 5; // final top K results to use for LLM
    int top_num_centroids = 4; // number of top K clusters need to wait to gather for each query
    int include_llm = false; // 0: not include, 1: include
    int retrieve_docs = true; // 0: not retrieve, 1: retrieve
    int my_id;

     /*** TODO: use a more efficient way to store the doc_contents cache */
    std::unordered_map<int, std::unordered_map<long, std::string>> doc_tables; // cluster_id -> emb_index -> pathname
    std::unordered_map<int, std::unordered_map<long, std::string>> doc_contents; // {cluster_id0:{ emb_index0: doc content0, ...}, cluster_id1:{...}, ...}
    /*** query_result: query_text -> QuerySearchResults 
     *   is a UDL local cache to store the cluster search results for queries that haven't notified the client 
     *   (due to not all cluster results are collected)
    */
    std::unordered_map<std::string, std::unique_ptr<QuerySearchResults>> query_results;
    /*** since same query may appear in different query batches from different clients. i.e. different people ask the same question
     *  query_request_tracker: query_text -> [(client_id, query_batch_id, finished_process), ..]
     *  query_request_tracker keep track of the batched query requests that requested the same type of query.
     *  This is used as a helper field for caching the query_results, and early reply to the client if the results are ready
     *  to delay garbage collection of the results for a query if there is still requesting qb.
    */
    std::unordered_map<std::string, std::vector<QueryRequestSource>> query_request_tracker;
    // std::unordered_map<std::string, std::future<std::string>> query_api_futures;

    bool load_doc_table(DefaultCascadeContextType* typed_ctxt, int cluster_id);
    bool get_doc(DefaultCascadeContextType* typed_ctxt, int cluster_id, long emb_index, std::string& res_doc);
    // Helper function to run LLM with top_k_docs asynchronously
    std::string run_llm_with_top_k_docs(const std::string& query_text, const std::vector<std::string>& top_k_docs, const std::string& model, const std::string& api_key);
    /*** Helper function to add intermediate result to udl cache
     *   check if the query existed in the cache
     *   and if all the results are collected for the query
     *   If all results are collected, return the top_k docs for the query
     *   If not all results collected, add this qb_id to the tracker
     */
    bool check_query_request_finished(const std::string& query_text, const uint32_t& client_id, const uint32_t& query_batch_id);
    void garbage_collect_query_results(const std::string& query_text, const uint32_t& client_id, const uint32_t& query_batch_id);
    void process_and_send_result(DefaultCascadeContextType* typed_ctxt, const std::string& query_text, uint32_t client_id, uint32_t query_batch_id, uint32_t qid);

    virtual void ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) override;

    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;
};

void initialize(ICascadeContext* ctxt);
std::shared_ptr<OffCriticalDataPathObserver> get_observer(ICascadeContext* ctxt, const nlohmann::json& config);
void release(ICascadeContext* ctxt);

}  // namespace cascade
}  // namespace derecho


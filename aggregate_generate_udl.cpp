#include <algorithm>
#include <memory>
#include <map>
#include <iostream>
#include <queue>
#include <unordered_map>


#include <cascade/user_defined_logic_interface.hpp>
#include <cascade/utils.hpp>
#include <cascade/cascade_interface.hpp>

#include "rag_utils.hpp"

namespace derecho{
namespace cascade{

#define MY_UUID     "11a3c123-3300-31ac-1866-0003ac330000"
#define MY_DESC     "UDL to aggregate the knn search results for each query from the clusters and run LLM with the query and its top_k closest docs."

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}


struct ClusterSearchResults{
     const std::string query_text;
     int total_cluster_num = 0;
     std::vector<int> collected_cluster_ids;
     bool collected_all_results = false;
     int top_k = 0;
     std::priority_queue<DocIndex> agg_cluster_results;

     ClusterSearchResults(const std::string& query_text, int total_cluster_num, int top_k): 
                         query_text(query_text), total_cluster_num(total_cluster_num), top_k(top_k) {}

     void add_cluster_result(int cluster_id, std::vector<DocIndex> cluster_results){
          if(std::find(collected_cluster_ids.begin(), collected_cluster_ids.end(), cluster_id) != collected_cluster_ids.end()){
               std::cerr << "ERROR: cluster id=" << cluster_id << " search result is already collected for query=" << this->query_text << std::endl;
               dbg_default_error("{} received same cluster={} result for query={}.", __func__, cluster_id, this->query_text);
               return;
          }
          this->collected_cluster_ids.push_back(cluster_id);
          // Add the cluster_results to the min_heap, and keep the size of the heap to be top_k
          for (const auto& doc_index : cluster_results) {
               if (static_cast<int>(agg_cluster_results.size()) < top_k) {
                    agg_cluster_results.push(doc_index);
               } else if (doc_index < agg_cluster_results.top()) {
                    agg_cluster_results.pop();
                    agg_cluster_results.push(doc_index);
               }
          }
     }

     bool is_all_results_collected(){
          if(static_cast<int>(collected_cluster_ids.size()) == total_cluster_num){
               collected_all_results = true;
          }
          return collected_all_results;
     }

};

class AggGenOCDPO: public DefaultOffCriticalDataPathObserver {

    int top_k = 5; // final top K results to use for LLM
    int top_num_centroids = 4; // number of top K clusters need to wait to gather for each query
    int include_llm = false; // 0: not include, 1: include

    int my_id; // the node id of this node; logging purpose


    virtual void ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) override {
        int cluster_id = 0; // TODO: change this
        std::cout << key_string << std::endl;
        std::string query_text;
        std::vector<DocIndex> cluster_results;
        try{
            deserialize_cluster_search_result_from_bytes(cluster_id, object.blob.bytes, object.blob.size, query_text, cluster_results);
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to deserialize the cluster searched result and query texts from the object." << std::endl;
            dbg_default_error("{}, Failed to deserialize the cluster searched result from the object.", __func__);
            return;
        }
        for (const auto& doc_index : cluster_results) {
            std::cout << doc_index << std::endl;
        }
        
    }

    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;
public:

    static void initialize() {
        if(!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<AggGenOCDPO>();
        }
    }
    static auto get() {
        return ocdpo_ptr;
    }

    void set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config){
        this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
        try{
            if (config.contains("top_num_centroids")) {
                this->top_num_centroids = config["top_num_centroids"].get<int>();
            }
            if (config.contains("final_top_k")) {
                this->top_k = config["final_top_k"].get<int>();
            }
            if (config.contains("include_llm")) {
                this->include_llm = config["include_llm"].get<bool>();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to convert top_num_centroids, top_k or include_llm from config" << std::endl;
            dbg_default_error("Failed to convert top_num_centroids, top_k or include_llm from config, at clusters_search_udl.");
        }
    }
};

std::shared_ptr<OffCriticalDataPathObserver> AggGenOCDPO::ocdpo_ptr;

void initialize(ICascadeContext* ctxt) {
    AggGenOCDPO::initialize();
}

std::shared_ptr<OffCriticalDataPathObserver> get_observer(
        ICascadeContext* ctxt,const nlohmann::json& config) {
    auto typed_ctxt = dynamic_cast<DefaultCascadeContextType*>(ctxt);
    std::static_pointer_cast<AggGenOCDPO>(AggGenOCDPO::get())->set_config(typed_ctxt,config);
    return AggGenOCDPO::get();
}

void release(ICascadeContext* ctxt) {
    // nothing to release
    return;
}

} // namespace cascade
} // namespace derecho

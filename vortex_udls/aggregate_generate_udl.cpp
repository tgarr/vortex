#include "aggregate_generate_udl.hpp"

namespace derecho {
namespace cascade {


QuerySearchResults::QuerySearchResults(const std::string& query_text, int total_cluster_num, int top_k) 
    : query_text(query_text), total_cluster_num(total_cluster_num), top_k(top_k) {}

bool QuerySearchResults::is_all_results_collected() {
    if(static_cast<int>(collected_cluster_ids.size()) == total_cluster_num){
        collected_all_results = true;
    }
    // print out the docIndex in the min heap for debugging
    std::priority_queue<DocIndex> tmp = agg_top_k_results;
    return collected_all_results;
}

void QuerySearchResults::add_cluster_result(int cluster_id, std::vector<DocIndex> cluster_results) {
    if(std::find(collected_cluster_ids.begin(), collected_cluster_ids.end(), cluster_id) != collected_cluster_ids.end()){
        return;
    }
    this->collected_cluster_ids.push_back(cluster_id);
    // Add the cluster_results to the min_heap, and keep the size of the heap to be top_k
    for (const auto& doc_index : cluster_results) {
        if (static_cast<int>(agg_top_k_results.size()) < top_k) {
            agg_top_k_results.push(doc_index);
        } else if (doc_index < agg_top_k_results.top()) {
            agg_top_k_results.pop();
            agg_top_k_results.push(doc_index);
        }
    }
}

QueryRequestSource::QueryRequestSource(uint32_t client_id, uint32_t query_batch_id, uint32_t qid, int total_cluster_num, int received_cluster_result_count, bool notified_client)
    : client_id(client_id), query_batch_id(query_batch_id), qid(qid), total_cluster_num(total_cluster_num), received_cluster_result_count(received_cluster_result_count), notified_client(notified_client) {}


void AggGenOCDPO::initialize() {
    if (!ocdpo_ptr) {
        ocdpo_ptr = std::make_shared<AggGenOCDPO>();
    }
}

std::shared_ptr<OffCriticalDataPathObserver> AggGenOCDPO::get() {
    return ocdpo_ptr;
}

void AggGenOCDPO::set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config) {
    this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
    try{
        if (config.contains("top_num_centroids")) this->top_num_centroids = config["top_num_centroids"].get<int>();
        if (config.contains("final_top_k")) this->top_k = config["final_top_k"].get<int>();
        if (config.contains("include_llm")) this->include_llm = config["include_llm"].get<bool>();
        if (config.contains("retrieve_docs")) this->retrieve_docs = config["retrieve_docs"].get<bool>();
        if (config.contains("openai_api_key")) this->openai_api_key = config["openai_api_key"].get<std::string>();
        if (config.contains("llm_model_name")) this->llm_model_name = config["llm_model_name"].get<std::string>();
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to convert top_num_centroids, top_k, include_llm, or retrieve_docs from config" << std::endl;
        dbg_default_error("Failed to convert top_num_centroids, top_k, include_llm, or retrieve_docs from config, at clusters_search_udl.");
    }
    // only create UDL level thread if using llm for async processing
    if (this->include_llm){
        this->notify_thread = std::make_unique<NotifyThread>(this->my_id, this);
        this->notify_thread->start(typed_ctxt);
    }
}


AggGenOCDPO::NotifyThread::NotifyThread(uint64_t thread_id, AggGenOCDPO* parent_udl)
    : my_thread_id(thread_id), parent(parent_udl), running(false) {}

void AggGenOCDPO::NotifyThread::start(DefaultCascadeContextType* typed_ctxt) {
    running = true;
    real_thread = std::thread(&NotifyThread::main_loop, this, typed_ctxt);
}

void AggGenOCDPO::NotifyThread::join() {
    if (real_thread.joinable()) {
        real_thread.join();
    }
}

void AggGenOCDPO::NotifyThread::signal_stop() {
    running = false;
    thread_signal.notify_all(); 
}


void AggGenOCDPO::NotifyThread::main_loop(DefaultCascadeContextType* typed_ctxt) {
    std::unique_lock<std::mutex> lock(parent->map_mutex, std::defer_lock);
    while (running) {
        lock.lock();
        parent->map_cv.wait(lock, [&] { 
            return parent->new_request || !parent->query_api_futures.empty() || !running; 
        });
        if (!running)
            break;

        for (auto it = parent->query_api_futures.begin(); it != parent->query_api_futures.end();) {
            auto& future = it->second;
            if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                // Future is ready; retrieve and process result
                std::string result = future.get();
                parent->query_results[it->first]->api_result = result;
                parent->process_result_and_notify_clients(typed_ctxt, it->first);
                it = parent->query_api_futures.erase(it);
            } else {
                ++it;
            }
        }
        lock.unlock();
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

bool AggGenOCDPO::load_doc_table(DefaultCascadeContextType* typed_ctxt, int cluster_id) {
    if (doc_tables.find(cluster_id) != doc_tables.end()) {
        return true;
    }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
    TimestampLogger::log(LOG_TAG_AGG_UDL_LOAD_EMB_DOC_MAP_START, my_id, 0, cluster_id);
#endif
    // 0. check the keys for this grouped embedding objects stored in cascade
    //    because of the message size, the map for one cluster may split into multiple chunks stored in Cascade
    bool stable = 1; 
    persistent::version_t version = CURRENT_VERSION;
    std::string table_prefix = "/rag/doc/emb_doc_map/cluster" + std::to_string(cluster_id);
    auto keys_future = typed_ctxt->get_service_client_ref().list_keys(version, stable, table_prefix);
    std::vector<std::string> map_obj_keys = typed_ctxt->get_service_client_ref().wait_list_keys(keys_future);
    if (map_obj_keys.empty()) {
        std::cerr << "Error: " << table_prefix <<" has no emb_doc_map object found in the KV store" << std::endl;
        dbg_default_error("[{}]at {}, Failed to find object prefix {} in the KV store.", gettid(), __func__, table_prefix);
        return -1;
    }
    std::priority_queue<std::string, std::vector<std::string>, CompareObjKey> filtered_keys = filter_exact_matched_keys(map_obj_keys, table_prefix);
    // 1. get the doc table for the cluster_id
    while(!filtered_keys.empty()){
        std::string map_obj_key = filtered_keys.top();
        filtered_keys.pop();
        auto get_query_results = typed_ctxt->get_service_client_ref().get(map_obj_key);
        auto& reply = get_query_results.get().begin()->second.get();
        if (reply.blob.size == 0) {
            std::cerr << "Error: failed to get the doc table for key=" << map_obj_key << std::endl;
            dbg_default_error("Failed to get the doc table for key={}.", map_obj_key);
            return false;
        }
        char* json_data = const_cast<char*>(reinterpret_cast<const char*>(reply.blob.bytes));
        std::string json_str(json_data, reply.blob.size);
        try{
            nlohmann::json doc_table_json = nlohmann::json::parse(json_str);
            for (const auto& [emb_index, pathname] : doc_table_json.items()) {
                this->doc_tables[cluster_id][std::stol(emb_index)] = "/rag/doc/" + std::to_string(pathname.get<int>());
            }
        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << "Error: load_doc_table JSON parse error: " << e.what() << std::endl;
            dbg_default_error("{}, JSON parse error: {}", __func__, e.what());
            return false;
        }
    }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING     
    TimestampLogger::log(LOG_TAG_AGG_UDL_LOAD_EMB_DOC_MAP_END, my_id, 0, cluster_id);
#endif
    return true;
}

bool AggGenOCDPO::get_doc(DefaultCascadeContextType* typed_ctxt, int cluster_id, long emb_index, std::string& res_doc) {
    if (doc_contents.find(cluster_id) != doc_contents.end()) {
        if (doc_contents[cluster_id].find(emb_index) != doc_contents[cluster_id].end()) {
            res_doc = doc_contents[cluster_id][emb_index];
            return true;
        }
    }
    bool loaded_doc_table = load_doc_table(typed_ctxt, cluster_id);
    if (!loaded_doc_table) {
        dbg_default_error("Failed to load the doc table for cluster_id={}.", cluster_id);
        return false;
    }
    if (doc_tables[cluster_id].find(emb_index) == doc_tables[cluster_id].end()) {
        std::cerr << "Error: failed to find the doc pathname for cluster_id=" << cluster_id << " and emb_id=" << emb_index << std::endl;
        dbg_default_error("Failed to find the doc pathname for cluster_id={} and emb_id={}, query={}.", cluster_id, emb_index);
        return false;
    }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
    TimestampLogger::log(LOG_TAG_AGG_UDL_LOAD_DOC_START, this->my_id, emb_index, cluster_id);
#endif 
    auto& pathname = doc_tables[cluster_id][emb_index];
    if(!retrieve_docs){
        res_doc = pathname;
        return true;
    }
    auto get_doc_results = typed_ctxt->get_service_client_ref().get(pathname);
    auto& reply = get_doc_results.get().begin()->second.get();
    if (reply.blob.size == 0) {
        std::cerr << "Error: failed to cascade get the doc content for pathname=" << pathname << std::endl;
        dbg_default_error("Failed to cascade get the doc content for pathname={}.", pathname);
        return false;
    }
    // parse the reply.blob.bytes to std::string
    char* doc_data = const_cast<char*>(reinterpret_cast<const char*>(reply.blob.bytes));
    std::string doc_str(doc_data, reply.blob.size);  /*** TODO: this is a copy, need to optimize */
    this->doc_contents[cluster_id][emb_index] = doc_str;
    res_doc = this->doc_contents[cluster_id][emb_index];
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
    TimestampLogger::log(LOG_TAG_AGG_UDL_LOAD_DOC_END, this->my_id, emb_index, cluster_id);
#endif
    return true;
}

bool AggGenOCDPO::get_topk_docs(DefaultCascadeContextType* typed_ctxt, std::string& query_text){
    auto& agg_top_k_results = this->query_results[query_text]->agg_top_k_results;
    auto& top_k_docs = this->query_results[query_text]->top_k_docs;
    top_k_docs.resize(agg_top_k_results.size());
    int i = agg_top_k_results.size();
    while (!agg_top_k_results.empty()) {
        i--;
        auto doc_index = agg_top_k_results.top();
        agg_top_k_results.pop();
        std::string res_doc;
        bool find_doc = get_doc(typed_ctxt,doc_index.cluster_id, doc_index.emb_id, res_doc);
        if (!find_doc) {
            dbg_default_error("Failed to get_doc for cluster_id={} and emb_id={}.", doc_index.cluster_id, doc_index.emb_id);
            return false;
        }
        top_k_docs[i] = std::move(res_doc);
    }
    this->query_results[query_text]->retrieved_top_k_docs = true;
    return true;
}

void AggGenOCDPO::async_run_llm_with_top_k_docs(const std::string& query_text) {
    auto& top_k_docs = query_results[query_text]->top_k_docs;
    auto& api_key = this->openai_api_key;
    auto& model = this->llm_model_name;
    
    query_api_futures[query_text] = std::async(std::launch::async, api_utils::run_gpt4o_mini, query_text, top_k_docs, model, api_key);
}

bool AggGenOCDPO::check_query_request_finished(const std::string& query_text, const uint32_t& client_id, const uint32_t& query_batch_id, const uint32_t& qid) {
    auto& tracked_query_request = query_request_tracker[query_text];
    for (auto& q_source : tracked_query_request) {
        if (q_source.client_id == client_id && q_source.query_batch_id == query_batch_id ) {
            q_source.received_cluster_result_count += 1;
            if (q_source.received_cluster_result_count > q_source.total_cluster_num) {
                std::cerr << "Error: received_cluster_result_count" << q_source.received_cluster_result_count << ">total_cluster_num=" << q_source.total_cluster_num << std::endl;
                assert (q_source.received_cluster_result_count <= q_source.total_cluster_num);
            }
            return q_source.notified_client;
        }
    }
    query_request_tracker[query_text].emplace_back(client_id, query_batch_id, qid, this->top_num_centroids, 1, false);
    return false;
}

void AggGenOCDPO::process_result_and_notify_clients(DefaultCascadeContextType* typed_ctxt, const std::string& query_text) {
    // 1. convert the query and top_k_docs to a json object
    nlohmann::json result_json;
    result_json["query"] = query_text;
    if (this->include_llm) {
        result_json["response"] = query_results[query_text]->api_result;
    }else{
        result_json["top_k_docs"] = query_results[query_text]->top_k_docs;
    }

    // 2. notify the result to clients that send the same query
    for (const auto& query_source : query_request_tracker[query_text]) {
        if (query_source.notified_client) {
            continue;
        }
        auto& client_id = query_source.client_id;
        auto& query_batch_id = query_source.query_batch_id;
        auto& qid = query_source.qid;
        // add the query_batch_id to the result_json for logging purposes
        result_json["query_batch_id"] = query_batch_id;
        std::string result_json_str = result_json.dump();
        Blob result_blob(reinterpret_cast<const uint8_t*>(result_json_str.c_str()), result_json_str.size());
        try {
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
            TimestampLogger::log(LOG_TAG_AGG_UDL_PUT_RESULT_START, client_id, query_batch_id, qid);
#endif
            std::string notification_pathname = "/rag/results/" + std::to_string(client_id);
            typed_ctxt->get_service_client_ref().notify(result_blob,notification_pathname,client_id);
            dbg_default_trace("[AggregateGenUDL] echo back to node {}", client_id);
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
            TimestampLogger::log(LOG_TAG_AGG_UDL_PUT_RESULT_END, client_id, query_batch_id, qid);
#endif
            // 3. (garbage collection) remove query and query_result from the cache
            garbage_collect_query_results(query_text, client_id, query_batch_id, qid);
        } catch (derecho::derecho_exception& ex) {
            std::cerr << "[AGGnotification ocdpo]: exception on notification:" << ex.what() << std::endl;
            dbg_default_error("[AGGnotification ocdpo]: exception on notification:{}", ex.what());
        }
    }
}

void AggGenOCDPO::garbage_collect_query_results(const std::string& query_text, const uint32_t& client_id, const uint32_t& query_batch_id, const uint32_t& qid) {
    auto& tracked_query_request = query_request_tracker[query_text];
    for (auto it = tracked_query_request.begin(); it != tracked_query_request.end(); ++it) {
        if (it->client_id == client_id && it->query_batch_id == query_batch_id && it->qid == qid) {
            it->notified_client = true;
            break;
        } 
    }
    /*** check if all the cluster search result of this query has been processed. 
        * If not, keep the query in the tracker longer, because this UDL will be triggered again by the remaining cluster search DULs results,
        * in which case, we would skip processing the query again.
    */
    bool all_finished = true;
    for (const auto& query_tracker : tracked_query_request) {
        if (!query_tracker.notified_client || query_tracker.received_cluster_result_count < query_tracker.total_cluster_num) {
            all_finished = false;
            break;
        }
    }
    if (all_finished) {
        query_results.erase(query_text);
        query_request_tracker.erase(query_text);
    }
}

void AggGenOCDPO::ocdpo_handler(const node_id_t sender, 
                                const std::string& object_pool_pathname, 
                                const std::string& key_string, 
                                const ObjectWithStringKey& object, 
                                const emit_func_t& emit, 
                                DefaultCascadeContextType* typed_ctxt, 
                                uint32_t worker_id) {
    // 0. parse the query information from the key_string
    int client_id, cluster_id, batch_id, qid;
    if (!parse_query_info(key_string, client_id, batch_id, cluster_id, qid)) {
        std::cerr << "Error: failed to parse the query_info from the key_string:" << key_string << std::endl;
        dbg_default_error("In {}, Failed to parse the query_info from the key_string:{}.", __func__, key_string);
        return;
    }
        
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
    int query_batch_id = batch_id * QUERY_BATCH_ID_MODULUS + qid % QUERY_BATCH_ID_MODULUS; // cast down qid for logging purpose
    TimestampLogger::log(LOG_TAG_AGG_UDL_START,client_id,query_batch_id,cluster_id);
#endif
    dbg_default_trace("[AggregateGenUDL] receive cluster search result from cluster{}.", cluster_id);
    std::string query_text;
    std::vector<DocIndex> cluster_results;
    // 1. deserialize the cluster searched result from the object
    try{
        deserialize_cluster_search_result_from_bytes(cluster_id, object.blob.bytes, object.blob.size, query_text, cluster_results);
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to deserialize the cluster searched result and query texts from the object." << std::endl;
        dbg_default_error("{}, Failed to deserialize the cluster searched result from the object.", __func__);
        return;
    }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
    TimestampLogger::log(LOG_TAG_AGG_UDL_FINISHED_DESERIALIZE, client_id, query_batch_id, cluster_id);
#endif
    new_request = true;
    std::unique_lock<std::mutex> lock(map_mutex);  // lock for map accessing
    /*** 1.1 If the query result has sent back to the client before, skip sending it again.
        * To handle the case where multiple different client send the same query 
        *  At aggregation step, we could use the local cache to directly send back to client what were collected before
        *  But UDL2 doesn't have caching and unaware of the same query, so it would recomputes KNN for the same query embedding and
        *  then trigger this UDL multiple time even after we send back the result to the client already. 
        *  This is to avoid sending the same result to the client multiple times.
    */
    if (query_results.find(query_text) == query_results.end()) {
        query_results[query_text] = std::make_unique<QuerySearchResults>(query_text, top_num_centroids, top_k);
        query_request_tracker[query_text] = std::vector<QueryRequestSource>();
    } 
    if (check_query_request_finished(query_text, client_id, query_batch_id, qid)) {
        // check if need to garbage clean the query results if all of its cluster_results have been processed
        garbage_collect_query_results(query_text, client_id, query_batch_id, qid); 
        goto cleanup;
    }
    // 2. add the cluster_results to the query_results and check if all results are collected
    query_results[query_text]->add_cluster_result(cluster_id, cluster_results);
    // 3. check if all cluster results are collected for this query
    if (!query_results[query_text]->is_all_results_collected()) {
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_TAG_AGG_UDL_END_NOT_FULLY_GATHERED, client_id, query_batch_id, cluster_id);
#endif
        goto cleanup;
    }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
    TimestampLogger::log(LOG_TAG_AGG_UDL_RETRIEVE_DOC_START, client_id, query_batch_id, cluster_id);
#endif
    // 4. All cluster results are collected. Retrieve the top_k docs contents
    if (!query_results[query_text]->retrieved_top_k_docs) {
        bool get_top_k_docs_success = get_topk_docs(typed_ctxt, query_text);
        if (!get_top_k_docs_success) {
            dbg_default_error("Failed to get top_k_docs for query_text={}.", query_text);
            goto cleanup;
        }
    }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
    TimestampLogger::log(LOG_TAG_AGG_UDL_RETRIEVE_DOC_END, client_id, query_batch_id, qid);
#endif
    // 5. run LLM with the query and its top_k closest docs
    if (include_llm) {
        async_run_llm_with_top_k_docs(query_text);
    } else {
    // 6. put the result to cascade and notify the client
        process_result_and_notify_clients(typed_ctxt, query_text);
    }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
    TimestampLogger::log(LOG_TAG_AGG_UDL_END, client_id, query_batch_id, qid);
#endif
cleanup:
    new_request = false;
    lock.unlock();
    map_cv.notify_one();
}



}  // namespace cascade
}  // namespace derecho
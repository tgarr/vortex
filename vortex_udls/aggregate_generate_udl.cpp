#include "aggregate_generate_udl.hpp"

namespace derecho {
namespace cascade {


QuerySearchResults::QuerySearchResults(const int& client_id, const int query_batch_id,
                    const std::string& query_text, const int& total_cluster_num, const int& top_k) 
    : client_id(client_id), query_batch_id(query_batch_id), query_text(query_text), 
        total_cluster_num(total_cluster_num), top_k(top_k) {}

bool QuerySearchResults::clusters_results_all_received() {
    if(static_cast<int>(collected_cluster_ids.size()) == total_cluster_num){
        return true;
    }
    return false;
}

void QuerySearchResults::add_cluster_result(int cluster_id, std::vector<DocIndex> cluster_results) {
    if(std::find(collected_cluster_ids.begin(), collected_cluster_ids.end(), cluster_id) != collected_cluster_ids.end()){
        dbg_default_warn("AggGenOCDPO, receiving repeated cluster's result, from cluster{}",cluster_id);
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


AggGenOCDPO::ProcessThread::ProcessThread(uint64_t thread_id, AggGenOCDPO* parent_udl)
    : my_thread_id(thread_id), parent(parent_udl), running(false) {}

void AggGenOCDPO::ProcessThread::start(DefaultCascadeContextType* typed_ctxt) {
    running = true;
    real_thread = std::thread(&ProcessThread::main_loop, this, typed_ctxt);
}

void AggGenOCDPO::ProcessThread::join() {
    if (real_thread.joinable()) {
        real_thread.join();
    }
}

void AggGenOCDPO::ProcessThread::signal_stop() {
    running = false;
    thread_signal.notify_all(); 
}

void AggGenOCDPO::ProcessThread::push_task(std::unique_ptr<queuedTask> task) {
    std::lock_guard<std::mutex> lock(thread_mtx);
    pending_tasks.push(std::move(task));
    thread_signal.notify_all();
}

void AggGenOCDPO::ProcessThread::notify_client(DefaultCascadeContextType* typed_ctxt, 
                                                std::string& result_json_str,
                                                const int& client_id) {
    
    Blob result_blob(reinterpret_cast<const uint8_t*>(result_json_str.c_str()), result_json_str.size());
    try {
        std::string notification_pathname = "/rag/results/" + std::to_string(client_id);
        typed_ctxt->get_service_client_ref().notify(result_blob,notification_pathname,client_id);
        dbg_default_trace("[AggregateGenUDL] echo back to node {}", client_id);
    } catch (derecho::derecho_exception& ex) {
        std::cerr << "[AGGnotification ocdpo]: exception on notification:" << ex.what() << std::endl;
        dbg_default_error("[AGGnotification ocdpo]: exception on notification:{}", ex.what());
    }
}

std::string AggGenOCDPO::ProcessThread::serialize_results(const int& client_id,
                                                        const std::vector<int>& result_ids, 
                                                        size_t start, size_t end){
    std::string result_json_str;
    // convert the query and top_k_docs to a json object
    nlohmann::json result_json = json::array();
    for (size_t i = start; i <= end; i++) {
        auto& query_result = this->query_results[client_id][result_ids[i]];
        const int& query_batch_id = query_result->query_batch_id;
        nlohmann::json result;
        result["query"] = query_result->query_text;
        result["top_k_docs"] = query_result->top_k_docs;
        result["query_batch_id"] = query_batch_id; // logging purpose
        result_json.push_back(result);

        int qid = query_result->query_batch_id % QUERY_BATCH_ID_MODULUS;
        TimestampLogger::log(LOG_TAG_AGG_UDL_PUT_RESULT_START, client_id, query_batch_id, qid);
    }
    result_json_str = result_json.dump(); 
    return result_json_str; // RVO
}

void AggGenOCDPO::ProcessThread::garbage_collect_query_results(std::map<int, std::vector<int>>& notified_results) {
    for (auto& client_results : notified_results) {
        for (auto& query_batch_id : client_results.second) {
            query_results[client_results.first].erase(query_batch_id);
        }
    }
}


void AggGenOCDPO::ProcessThread::notify_clients_in_batch(DefaultCascadeContextType* typed_ctxt, 
                        std::map<int, std::vector<int>>& results_to_notify){
    for (const auto& [client_id, client_results] : results_to_notify) {
        size_t total_num_replies = client_results.size();
        size_t start_index = 0;
        size_t end_index;
        while (start_index < total_num_replies) {
            end_index = std::min(start_index + MAX_NUM_REPLIES_PER_NOTIFICATION_MESSAGE, total_num_replies) - 1;
            std::string result_json_str = serialize_results(client_id, client_results, start_index, end_index);
            notify_client(typed_ctxt, result_json_str, client_id);
            start_index = end_index + 1;
        }
    }
    // garbage collect the query_results
    garbage_collect_query_results(results_to_notify);
}

bool AggGenOCDPO::ProcessThread::get_topk_docs(DefaultCascadeContextType* typed_ctxt, 
                                            const int& client_id, const int& query_batch_id){
    auto& agg_top_k_results = this->query_results[client_id][query_batch_id]->agg_top_k_results;
    auto& top_k_docs = this->query_results[client_id][query_batch_id]->top_k_docs;
    top_k_docs.resize(agg_top_k_results.size());
    int i = agg_top_k_results.size();
    while (!agg_top_k_results.empty()) {
        i--;
        auto doc_index = agg_top_k_results.top();
        agg_top_k_results.pop();
        std::string res_doc;
        bool find_doc = parent->get_doc(typed_ctxt,doc_index.cluster_id, doc_index.emb_id, res_doc);
        if (!find_doc) {
            dbg_default_error("Failed to get_doc for cluster_id={} and emb_id={}.", doc_index.cluster_id, doc_index.emb_id);
            return false;
        }
        top_k_docs[i] = std::move(res_doc);
    }
    return true;
}

void AggGenOCDPO::ProcessThread::add_to_query_results(queuedTask* task_ptr) {
    if (query_results.find(task_ptr->client_id) == query_results.end()) {
        query_results[task_ptr->client_id] = std::unordered_map<int, std::unique_ptr<QuerySearchResults>>();
    }
    if (query_results[task_ptr->client_id].find(task_ptr->query_batch_id) == query_results[task_ptr->client_id].end()) {
        query_results[task_ptr->client_id][task_ptr->query_batch_id] = std::make_unique<QuerySearchResults>(
                                                                    task_ptr->client_id, task_ptr->query_batch_id,
                                                                    task_ptr->query_text, parent->top_num_centroids, parent->top_k);
    }
    query_results[task_ptr->client_id][task_ptr->query_batch_id]->add_cluster_result(task_ptr->cluster_id, task_ptr->cluster_results);
}


bool AggGenOCDPO::ProcessThread::process_task(DefaultCascadeContextType* typed_ctxt, queuedTask* task_ptr){
    // 1. add the cluster_results to the query_results and check if all results are collected
    add_to_query_results(task_ptr);
    // 2. check if all cluster results are collected for this query
    if (!query_results[task_ptr->client_id][task_ptr->query_batch_id]->clusters_results_all_received()) {
        TimestampLogger::log(LOG_TAG_AGG_UDL_END_NOT_FULLY_GATHERED, task_ptr->client_id, task_ptr->query_batch_id, task_ptr->cluster_id);
        return false;
    }
    TimestampLogger::log(LOG_TAG_AGG_UDL_RETRIEVE_DOC_START, task_ptr->client_id, task_ptr->query_batch_id, task_ptr->cluster_id);
    // 3. All cluster results are collected. Retrieve the top_k docs contents
    bool get_top_k_docs_success = this->get_topk_docs(typed_ctxt, task_ptr->client_id, task_ptr->query_batch_id);
    if (!get_top_k_docs_success) {
        dbg_default_error("Failed to get top_k_docs for query_text={}.", task_ptr->query_text);
        return false;
    }
    TimestampLogger::log(LOG_TAG_AGG_UDL_RETRIEVE_DOC_END, task_ptr->client_id, task_ptr->query_batch_id, task_ptr->qid);
    return true;
}

void AggGenOCDPO::ProcessThread::process_tasks(DefaultCascadeContextType* typed_ctxt, 
                                            std::vector<std::unique_ptr<queuedTask>>& tasks,
                                            std::map<int, std::vector<int>>& results_to_notify){
    for (auto& task_ptr : tasks) {
        bool collected_all_results = process_task(typed_ctxt, task_ptr.get());
        if (collected_all_results) {
            results_to_notify[task_ptr->client_id].push_back(task_ptr->query_batch_id);
        }
    }
}

void AggGenOCDPO::ProcessThread::main_loop(DefaultCascadeContextType* typed_ctxt) {
    auto batch_time = std::chrono::microseconds(parent->batch_time_us);
    auto wait_start = std::chrono::steady_clock::now();
    std::vector<std::unique_ptr<queuedTask>> tasks;
    tasks.reserve(parent->max_batch_size);

    while (running) {
        std::unique_lock<std::mutex> lock(thread_mtx);
        if (this->pending_tasks.size() < parent->min_batch_size ){
            thread_signal.wait_for(lock, batch_time);
        }
        if (!running)
            break;
        auto now = std::chrono::steady_clock::now();
        auto pending_count = this->pending_tasks.size();
        auto current_batch_count = 0;
        // move out the pending tasks in batch to be processed
        if (pending_count >= parent->min_batch_size || (now - wait_start) >= batch_time) {
            current_batch_count = std::min(pending_count, static_cast<size_t>(parent->max_batch_size));
            wait_start = now;
            while (!this->pending_tasks.empty() && tasks.size() < current_batch_count) {
                tasks.push_back(std::move(this->pending_tasks.front()));
                this->pending_tasks.pop();
            }
        }
        lock.unlock();

        // process pending tasks in batch
        std::map<int, std::vector<int>> results_to_notify;
        if (current_batch_count != 0) {
            process_tasks(typed_ctxt, tasks, results_to_notify);
            tasks.clear();
        }
        // batch the queries for each client to send notifications
        notify_clients_in_batch(typed_ctxt, results_to_notify);
    }
}


bool AggGenOCDPO::load_doc_table(DefaultCascadeContextType* typed_ctxt, int cluster_id) {
    if (doc_tables.find(cluster_id) != doc_tables.end()) {
        return true;
    }
    TimestampLogger::log(LOG_TAG_AGG_UDL_LOAD_EMB_DOC_MAP_START, my_id, 0, cluster_id);
    // check the keys for this grouped embedding objects stored in cascade
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
    // get the doc table objects and save to local cache
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
    TimestampLogger::log(LOG_TAG_AGG_UDL_LOAD_EMB_DOC_MAP_END, my_id, 0, cluster_id);
    return true;
}

bool AggGenOCDPO::get_doc(DefaultCascadeContextType* typed_ctxt, int cluster_id, long emb_index, std::string& res_doc) {
    std::shared_lock<std::shared_mutex> read_lock(doc_cache_mutex);
    if (doc_contents.find(cluster_id) != doc_contents.end()) {
        if (doc_contents[cluster_id].find(emb_index) != doc_contents[cluster_id].end()) {
            res_doc = doc_contents[cluster_id][emb_index];
            return true;
        }
    }
    read_lock.unlock();
    // doc not found in the cache, load the emb_index->doc table and content, and cache it
    std::unique_lock<std::shared_mutex> write_lock(doc_cache_mutex);
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
    TimestampLogger::log(LOG_TAG_AGG_UDL_LOAD_DOC_START, this->my_id, emb_index, cluster_id);
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
    std::string doc_str(doc_data, reply.blob.size);  
    this->doc_contents[cluster_id][emb_index] = doc_str;
    res_doc = this->doc_contents[cluster_id][emb_index];
    TimestampLogger::log(LOG_TAG_AGG_UDL_LOAD_DOC_END, this->my_id, emb_index, cluster_id);
    return true;
}


void AggGenOCDPO::ocdpo_handler(const node_id_t sender, 
                                const std::string& object_pool_pathname, 
                                const std::string& key_string, 
                                const ObjectWithStringKey& object, 
                                const emit_func_t& emit, 
                                DefaultCascadeContextType* typed_ctxt, 
                                uint32_t worker_id) {
    // 1. parse the query information from the key_string
    std::unique_ptr<queuedTask> task_ptr = std::make_unique<queuedTask>();
    if (!parse_query_info(key_string, task_ptr->client_id, task_ptr->batch_id, task_ptr->cluster_id, task_ptr->qid)) {
        std::cerr << "Error: failed to parse the query_info from the key_string:" << key_string << std::endl;
        dbg_default_error("In {}, Failed to parse the query_info from the key_string:{}.", __func__, key_string);
        return;
    }
    task_ptr->query_batch_id = task_ptr->batch_id * QUERY_BATCH_ID_MODULUS + task_ptr->qid % QUERY_BATCH_ID_MODULUS; // cast down qid for logging purpose
    TimestampLogger::log(LOG_TAG_AGG_UDL_START,task_ptr->client_id,task_ptr->query_batch_id,task_ptr->cluster_id);
    dbg_default_trace("[AggregateGenUDL] receive cluster search result from cluster{}.", task_ptr->cluster_id);
    // 2. deserialize the cluster searched result from the object
    try{
        deserialize_cluster_search_result_from_bytes(task_ptr->cluster_id, object.blob.bytes, object.blob.size, task_ptr->query_text, task_ptr->cluster_results);
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to deserialize the cluster searched result and query texts from the object." << std::endl;
        dbg_default_error("{}, Failed to deserialize the cluster searched result from the object.", __func__);
        return;
    }
    TimestampLogger::log(LOG_TAG_AGG_UDL_FINISHED_DESERIALIZE, task_ptr->client_id, task_ptr->query_batch_id, task_ptr->cluster_id);
    // if multithread, could allocate based on qid(the hashing of query_text from UDL2) % num_threads
    process_thread->push_task(std::move(task_ptr));
    // TODO: add a logger here after push?
}

void AggGenOCDPO::start_threads(DefaultCascadeContextType* typed_ctxt) {
    uint64_t thread_id = 0;
    /*** Note: current implementation only have one thread to proces queue,
     *   Could add more thread to process tasks, 
     *   but need to make sure the same query_text is processed by the same thread  
     */
    process_thread = std::make_unique<ProcessThread>(thread_id, this);
    process_thread->start(typed_ctxt);
}

void AggGenOCDPO::set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config) {
    this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
    try{
        if (config.contains("top_num_centroids")) this->top_num_centroids = config["top_num_centroids"].get<int>();
        if (config.contains("final_top_k")) this->top_k = config["final_top_k"].get<int>();
        if (config.contains("retrieve_docs")) this->retrieve_docs = config["retrieve_docs"].get<bool>();
        if (config.contains("batch_time_us")) this->batch_time_us = config["batch_time_us"].get<int>();
        if (config.contains("min_batch_size")) this->min_batch_size = config["min_batch_size"].get<int>();
        if (config.contains("max_batch_size")) this->max_batch_size = config["max_batch_size"].get<int>();
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to convert top_num_centroids, top_k, include_llm, or retrieve_docs from config" << std::endl;
        dbg_default_error("Failed to convert top_num_centroids, top_k, include_llm, or retrieve_docs from config, at clusters_search_udl.");
    } 
    this->start_threads(typed_ctxt);
}  

void AggGenOCDPO::shutdown() {
    process_thread->signal_stop();
    process_thread->join();
}


}  // namespace cascade
}  // namespace derecho
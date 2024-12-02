#include <iostream>
#include <thread>
#include <chrono>

// #include "search_worker.hpp"
#include "clusters_search_udl.hpp"


namespace derecho{
namespace cascade{


ClustersSearchOCDPO::ClusterSearchWorker::ClusterSearchWorker(uint64_t thread_id, 
                                        ClustersSearchOCDPO* parent_udl)
        : my_thread_id(thread_id), parent(parent_udl), running(false){
    use_shadow_flag = 0;
    query_buffer = std::make_unique<queryQueue>(parent->emb_dim);
    shadow_query_buffer = std::make_unique<queryQueue>(parent->emb_dim);
}


void ClustersSearchOCDPO::ClusterSearchWorker::construct_new_keys(std::vector<std::string>& new_keys,
                                        std::vector<std::string>::const_iterator keys_begin,
                                        std::vector<std::string>::const_iterator keys_end,
                                        std::vector<std::string>::const_iterator queries_begin,
                                        std::vector<std::string>::const_iterator queries_end) {
    auto key_it = keys_begin;
    auto query_it = queries_begin;

    while (key_it != keys_end ) {
        try {
            uint8_t digest[32];
            openssl::Hasher sha256(openssl::DigestAlgorithm::SHA256);
            const char* query_cstr = query_it->c_str();
            sha256.hash_bytes(query_cstr, strlen(query_cstr), digest);
            // Convert digest to a hexadecimal string
            std::ostringstream oss;
            for (int i = 0; i < 32; ++i) {
                // Output each byte as a decimal value (0-255) without any leading zeros
                oss << std::dec << static_cast<int>(digest[i]);
            }
            std::string hashed_query = oss.str();
            // Construct the new key
            new_keys.emplace_back(*key_it + "_qid" + hashed_query);
        } catch (openssl::openssl_error& ex) {
            dbg_default_error(
                "Unable to compute SHA256 of typename. string = {}, exception name = {}", *query_it, ex.what());
            throw;
        }
        ++key_it;
        ++query_it;
    }
    // Sanity check
    if (key_it != keys_end || query_it != queries_end) {
        dbg_default_error("Mismatched iterator ranges in construct_new_keys");
        throw std::runtime_error("Mismatched iterator ranges in construct_new_keys");
    }
}


/*** TODO: could batch the results, which are going to be send to the same shard */
void ClustersSearchOCDPO::ClusterSearchWorker::emit_results(DefaultCascadeContextType* typed_ctxt,
                                                    const std::vector<std::string>& new_keys,
                                                    const std::vector<std::string>& query_list,
                                                    long* I, float* D, size_t start_idx, size_t batch_size) {
    for (size_t k = 0; k < batch_size; ++k) {
        ObjectWithStringKey obj;
        obj.key = std::string(EMIT_AGGREGATE_PREFIX) + "/" + new_keys[k];
        std::string query_emit_content = serialize_cluster_search_result(parent->top_k, I, D, k, query_list[start_idx + k]);
        obj.blob = Blob(reinterpret_cast<const uint8_t*>(query_emit_content.c_str()), query_emit_content.size());
        // logging purpose
        // int client_id = -1, query_batch_id = -1;
        // parse_batch_id(obj.key, client_id, query_batch_id);
        // TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_EMIT_START, client_id, query_batch_id, this->cluster_id);
        // emit result
        typed_ctxt->get_service_client_ref().put_and_forget(obj,true);
    }
}

void ClustersSearchOCDPO::ClusterSearchWorker::run_cluster_search_and_emit(DefaultCascadeContextType* typed_ctxt, 
                                                                            queryQueue* query_buffer) {
    // Check if the cluster search index is initialized
    if (!parent->cluster_search_index->initialized_index.load()){
        if(!parent->check_and_retrieve_cluster_index(typed_ctxt)){
            return;
        }
    }

    size_t num_queries = query_buffer->count_queries();
    for (size_t start_idx = 0; start_idx < num_queries; start_idx += parent->max_batch_size) {
        size_t cur_batch_size = std::min(static_cast<size_t>(parent->max_batch_size), num_queries - start_idx);
        // 1. perform batched ANN search
        long* I = new long[parent->top_k * cur_batch_size];
        float* D = new float[parent->top_k * cur_batch_size];
        parent->cluster_search_index->search(cur_batch_size, query_buffer->query_embs + start_idx * parent->emb_dim, parent->top_k, D, I);
        if (!I || !D) {
            dbg_default_error("Failed to batch search for cluster: {}", parent->cluster_id);
            delete[] I;
            delete[] D;
            return;
        }

        // 2. Generate emit keys for this batch
        std::vector<std::string> new_keys;
        construct_new_keys(new_keys,
            query_buffer->query_keys.begin() + start_idx,
            query_buffer->query_keys.begin() + start_idx + cur_batch_size,
            query_buffer->query_list.begin() + start_idx,
            query_buffer->query_list.begin() + start_idx + cur_batch_size
        );
        // 3. Emit results 
        emit_results(typed_ctxt, new_keys, query_buffer->query_list, I, D, start_idx, cur_batch_size);

        delete[] I;
        delete[] D;
    }
}



// check number of pending queries
bool ClustersSearchOCDPO::ClusterSearchWorker::enough_pending_queries(int num){
    if (use_shadow_flag == 1 && shadow_query_buffer->count_queries() >= num) {
        return true;
    } else if (use_shadow_flag == 0 && query_buffer->count_queries() >= num) {
        return true;
    }
    return false;
}

void ClustersSearchOCDPO::ClusterSearchWorker::main_loop(DefaultCascadeContextType* typed_ctxt) {
    auto batch_time = std::chrono::microseconds(parent->batch_time_us);
    while (running) {
        std::unique_lock<std::mutex> query_buff_lock(query_buffer_mutex); 
        // check if there are queries to be processed on the current processing buffer
        query_buffer_cv.wait_for(query_buff_lock, std::chrono::microseconds(parent->batch_time_us),[this]() {
            return this->enough_pending_queries(parent->min_batch_size); ; 
        });
        if (!running) break;
        if ( !this->enough_pending_queries(1)) {
            continue;
        }

        // TODO: if push_queries happens too frequent, add priority to the main_loop to process the queries
        use_shadow_flag ^= 1; // flip the shadow flag, so the push_thread can start a new buffer
        query_buff_lock.unlock();
        
        if (!running) break;

        queryQueue* cur_query_buffer = nullptr;
        if (use_shadow_flag == 1) {
            cur_query_buffer = query_buffer.get();
        } else {
            cur_query_buffer = shadow_query_buffer.get();
        }
        run_cluster_search_and_emit(typed_ctxt, cur_query_buffer);
    
        cur_query_buffer->reset();
    }
}

void ClustersSearchOCDPO::ClusterSearchWorker::push_to_query_buffer(int cluster_id, 
                                                                    const Blob& blob, 
                                                                    const std::string& key) {
    // 1. get the query embeddings from the object
    float* data;
    uint32_t nq;
    std::vector<std::string> query_list;
    try{
        deserialize_embeddings_and_quries_from_bytes(blob.bytes, blob.size, nq, parent->emb_dim, data, query_list);
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to deserialize the query embeddings and query texts from the object." << std::endl;
        dbg_default_error("{}, Failed to deserialize the query embeddings and query texts from the object.", __func__);
        return;
    }
    // 2. add the queries to the queueing batch
    std::unique_lock<std::mutex> lock(query_buffer_mutex);
    // this step incurs a copy of the query embedding float[], to make it aligned with the other embeddings in the buffer for batching
    if (use_shadow_flag == 1) {
        shadow_query_buffer->add_queries(std::move(query_list), key, data, parent->emb_dim, nq);
    } else {
        query_buffer->add_queries(std::move(query_list), key, data, parent->emb_dim, nq);
    }
    query_buffer_cv.notify_one();
}

void ClustersSearchOCDPO::ClusterSearchWorker::start(DefaultCascadeContextType* typed_ctxt) {
    running = true;
    real_thread = std::thread(&ClusterSearchWorker::main_loop, this, typed_ctxt);
}

void ClustersSearchOCDPO::ClusterSearchWorker::join() {
    if (real_thread.joinable()) {
        real_thread.join();
    }
}

void ClustersSearchOCDPO::ClusterSearchWorker::signal_stop() {
    running = false;
    query_buffer_cv.notify_all();
}


bool ClustersSearchOCDPO::check_and_retrieve_cluster_index(DefaultCascadeContextType* typed_ctxt){
    // Acquire a unique lock to modify the cluster search index
    std::unique_lock<std::shared_mutex> write_lock(cluster_search_index_mutex);
    // Double-check if the cluster was inserted by another thread
    if (cluster_search_index->initialized_index.load()) {
        return true;
    } 
    TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_LOADEMB_START, my_id, this->cluster_id, 0);
    std::string cluster_prefix = CLUSTER_EMB_OBJECTPOOL_PREFIX + std::to_string(this->cluster_id);
    int filled_cluster_embs = cluster_search_index->retrieve_grouped_embeddings(cluster_prefix, typed_ctxt);
    if (filled_cluster_embs == -1) {
        std::cerr << "Error: failed to fill the cluster embeddings in cache" << std::endl;
        dbg_default_error("Failed to fill the cluster embeddings in cache, at clusters_search_udl.");
        return false;
    }
    int initialized = cluster_search_index->initialize_groupped_embeddings_for_search();
    if (initialized == -1) {
        std::cerr << "Error: failed to initialize the index for the cluster embeddings" << std::endl;
        dbg_default_error("Failed to initialize the index for the cluster embeddings, at clusters_search_udl.");
        return false;
    }
    TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_LOADEMB_END, my_id, this->cluster_id, 0);

    return true;
}
    
void ClustersSearchOCDPO::ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) {
    // TODO timestamp logging need to be revisited

    /*** Note: this object_pool_pathname is trigger pathname prefix: /rag/emb/clusteres_search instead of /rag/emb, i.e. the objp name***/
    dbg_default_trace("[Clusters search ocdpo]: I({}) received an object from sender:{} with key={}", worker_id, sender, key_string);
    
    // 0. parse the key, get the cluster ID
    uint64_t parsed_cluster_id = parse_cluster_id(key_string);

    // TODO do we need to enforce the cluster ID here ? it could work for any cluster
    if (this->cluster_id == -1) {
        this->cluster_id = parsed_cluster_id;
    } else if (this->cluster_id != parsed_cluster_id) {
        std::cerr << "Error: cluster ID mismatched" << std::endl;
        dbg_default_error("Cluster ID mismatched, at clusters_search_udl.");
        return;
    }

    TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_START,my_id,0,parsed_cluster_id);
    
    std::unique_ptr<EmbeddingQueryBatchManager> batch_manager = std::make_unique<EmbeddingQueryBatchManager>(object.blob.bytes,object.blob.size,emb_dim,false);
    
    // 1. Add queries to the to the appropriate queue in the next thread
    this->cluster_search_threads[next_thread]->push_to_query_buffer(parsed_cluster_id, std::move(batch_manager));
    next_thread = (next_thread + 1) % this->num_threads; // cycle
    
    dbg_default_trace("[Cluster search ocdpo]: PUT {} to active queue on thread {}.", key_string, next_thread);
}


void ClustersSearchOCDPO::start_threads(DefaultCascadeContextType* typed_ctxt) {
    for (int thread_id = 0; thread_id < this->num_threads; thread_id++) {
        cluster_search_threads.emplace_back(std::make_unique<ClusterSearchWorker>(static_cast<uint64_t>(thread_id), this));
    }
    for (auto& cluster_search_thread : cluster_search_threads) {
        cluster_search_thread->start(typed_ctxt);
    }
}


void ClustersSearchOCDPO::set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config){
    this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
    try{
        if (config.contains("emb_dim")) {
            this->emb_dim = config["emb_dim"].get<int>();
        }
        if (config.contains("top_k")) {
            this->top_k = config["top_k"].get<uint32_t>();
        }
        if (config.contains("faiss_search_type")) {
            this->faiss_search_type = config["faiss_search_type"].get<int>();
        }
        if (config.contains("batch_time_us")) {
            this->batch_time_us = config["batch_time_us"].get<uint32_t>();
        }
        if (config.contains("min_batch_size")) {
            this->min_batch_size = config["min_batch_size"].get<uint32_t>();
        }
        if (config.contains("max_batch_size")) {
            this->max_batch_size = config["max_batch_size"].get<uint32_t>();
        }
        if (config.contains("min_process_batch_size")) {
            this->min_process_batch_size = config["min_process_batch_size"].get<uint32_t>();
        }
        if (config.contains("max_process_batch_size")) {
            this->max_process_batch_size = config["max_process_batch_size"].get<uint32_t>();
        }
        if (config.contains("process_batch_time_us")) {
            this->process_batch_time_us = config["process_batch_time_us"].get<uint32_t>();
        }
        // Currenlty only support multithreading for hnswlib search, as GPU flat search requires cuda streams for host side parallelism
        if (config.contains("num_threads") && static_cast<GroupedEmbeddingsForSearch::SearchType>(this->faiss_search_type) == GroupedEmbeddingsForSearch::SearchType::HnswlibCpuSearch) {
            this->num_threads = config["num_threads"].get<int>();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to convert emb_dim or top_k from config" << std::endl;
        dbg_default_error("Failed to convert emb_dim or top_k from config, at clusters_search_udl.");
    }
    this->cluster_search_index = std::make_shared<GroupedEmbeddingsForSearch>(this->faiss_search_type, this->emb_dim);
    this->start_threads(typed_ctxt);
}

void ClustersSearchOCDPO::shutdown() {
    std::unique_lock<std::shared_mutex> lock(cluster_search_index_mutex);
    // Clean up index resources
    cluster_search_index->reset();  
    for (auto& cluster_search_thread : cluster_search_threads) {
        if (cluster_search_thread) {
            cluster_search_thread->signal_stop();
            cluster_search_thread->join();
        }
    }
}




} // namespace cascade
} // namespace derecho

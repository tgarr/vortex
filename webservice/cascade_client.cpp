
#include "cascade_client.hpp"

VortexCascadeClient::VortexCascadeClient(){
}

VortexCascadeClient::~VortexCascadeClient(){
    client_thread->signal_stop();
    client_thread->join();

    for(auto &t : notification_threads){
        t.signal_stop();
    }
    
    for(auto &t : notification_threads){
        t.join();
    }

    // print batching statistics
    std::cout << "batching statistics:" << std::endl;
    std::vector<uint64_t> values;
    values.reserve(client_thread->batch_size.size());
    double sum = 0.0;
    for(const auto& [batch_id, sz] : client_thread->batch_size){
        values.push_back(sz);
        sum += sz;
    }

    double avg = sum / client_thread->batch_size.size();
    std::sort(values.begin(),values.end());
    auto min = values.front();
    auto max = values.back();
    auto median = values[values.size()/2];
    auto p95 = values[(uint64_t)(values.size()*0.95)];

    std::cout << "  avg: " << avg << std::endl;
    std::cout << "  median: " << median << std::endl;
    std::cout << "  min: " << min << std::endl;
    std::cout << "  max: " << max << std::endl;
    std::cout << "  p95: " << p95 << std::endl;
}

void VortexCascadeClient::setup(uint64_t batch_min_size,uint64_t batch_max_size,uint64_t batch_time_us,uint64_t emb_dim,uint64_t num_result_threads){
    this->emb_dim = emb_dim;
    this->num_result_threads = num_result_threads;

    // Prepare for the notification by creating object pool for results to store
    std::string result_pool_name = "/rag/results/" + std::to_string(this->my_id);
    std::cout << "  creating object pool for receiving results: " << result_pool_name << std::endl;
    auto res = capi.template create_object_pool<UDLS_SUBGROUP_TYPE>(result_pool_name,UDL3_SUBGROUP_INDEX,HASH,{});
    for (auto& reply_future:res.get()) {
        reply_future.second.get(); // wait for the object pool to be created
    }

    // Register notification for this object pool
    std::cout << "  registering notification handler ... " << result_pool_name << std::endl;
    bool ret = capi.register_notification_handler(
            [&](const Blob& result){
                notification_threads[next_thread].push_result(result);
                next_thread = (next_thread + 1) % this->num_result_threads;
                return true;
            }, result_pool_name);

    // Establish connections to all server nodes running UDL3
    auto shards = capi.get_subgroup_members<UDLS_SUBGROUP_TYPE>(UDL3_SUBGROUP_INDEX);
    std::cout << "  pre-establishing connections with all nodes in " << shards.size() << " shards ..." << std::endl;
    ObjectWithStringKey obj;
    obj.key = "establish_connection";
    std::string control_value = "establish";
    obj.blob = Blob(reinterpret_cast<const uint8_t*>(control_value.c_str()), control_value.size());
    uint32_t i = 0;
    for(auto& shard : shards){
        for(int j=0;j<shard.size();j++){
            // each iteration reaches a different node in the shard due to the round robin policy
            auto res = capi.template put<UDLS_SUBGROUP_TYPE>(obj, UDL3_SUBGROUP_INDEX, i, true);
            for (auto& reply_future:res.get()) {
                reply_future.second.get(); // wait for the object has been put
            }
        }
        i++;
    }

    // start notification threads
    for(uint64_t i=0;i<num_result_threads;i++){
        notification_threads.emplace_back(this);
    }

    for(auto &t : notification_threads){
        t.start();
    }

    // start client thread
    client_thread = new ClientThread(batch_min_size,batch_max_size,batch_time_us,emb_dim);
    client_thread->start();
}

query_id_t VortexCascadeClient::next_query_id(){
    std::unique_lock<std::mutex> lock(query_id_mtx);
    query_id_t query_id = (my_id << 48) | query_count;
    query_count++;
    return query_id;
}

std::future<std::vector<std::string>> VortexCascadeClient::query(const std::string& query,const float* query_emb){
    query_id_t query_id = VortexCascadeClient::next_query_id();
    //queued_query_t new_query(query_id,query,query_emb);
    queued_query_t new_query(query_id,std::to_string(query_id),query_emb); // XXX for our experiments

    std::unique_lock<std::shared_mutex> lock(result_mutex);
    auto& prom = result[query_id];
    lock.unlock();

    client_thread->push_query(new_query);
    return prom.get_future();
}

void VortexCascadeClient::result_received(nlohmann::json &all_results_json){
    for (const auto& result_json : all_results_json) {
        if (result_json.count("query") == 0 || result_json.count("top_k_docs") == 0) {
            std::cerr << "Result JSON does not contain query or top_k_docs." << std::endl;
            continue;
        }

        std::string query_text(std::move(result_json["query"]));
        query_id_t query_id = std::stoll(query_text); // XXX for our experiments

        std::shared_lock<std::shared_mutex> lock(result_mutex);
        auto& prom = result[query_id];
        lock.unlock();
       
        std::vector<std::string> values(std::move(result_json["top_k_docs"])); 
        for(auto& item : values){
            item = item.substr(item.find_last_of('/') + 1);
        }

        prom.set_value(std::move(values));
    }
}

// client thread methods

VortexCascadeClient::ClientThread::ClientThread(uint64_t batch_min_size,uint64_t batch_max_size,uint64_t batch_time_us,uint64_t emb_dim){
    this->batch_min_size = batch_min_size;
    this->batch_max_size = batch_max_size;
    this->batch_time_us = batch_time_us;
    this->emb_dim = emb_dim;
}

void VortexCascadeClient::ClientThread::push_query(queued_query_t &queued_query){
    std::unique_lock<std::mutex> lock(thread_mtx);
    query_queue.push(queued_query);
    thread_signal.notify_all();
}

void VortexCascadeClient::ClientThread::signal_stop(){
    std::unique_lock<std::mutex> lock(thread_mtx);
    running = false;
    thread_signal.notify_all();
}

// helper function
void inline build_batch_string(std::string& batch_string,queued_query_t* to_send,uint64_t send_count,uint64_t emb_dim){
    // TODO this seems inefficient, but since it is not the bottleneck, we can leave it this way for now

    // create an bytes object by concatenating: num_queries + float array of embeddings + list of query_text
    batch_string.reserve(sizeof(uint32_t) + (send_count * sizeof(float) * emb_dim) + (send_count * 200));
    
    uint32_t num_queries = static_cast<uint32_t>(send_count);
    std::string nq_bytes(4, '\0');
    nq_bytes[0] = (num_queries >> 24) & 0xFF;
    nq_bytes[1] = (num_queries >> 16) & 0xFF;
    nq_bytes[2] = (num_queries >> 8) & 0xFF;
    nq_bytes[3] = num_queries & 0xFF;
    batch_string += nq_bytes;

    std::vector<std::string> query_list;
    query_list.reserve(num_queries);
    for(uint32_t i=0;i<num_queries;i++){
        batch_string.append(reinterpret_cast<const char*>(std::get<2>(to_send[i])),sizeof(float) * emb_dim);
        query_list.push_back(std::get<1>(to_send[i]));
    }

    batch_string += nlohmann::json(query_list).dump();
}

void VortexCascadeClient::ClientThread::main_loop(){
    if(!running) return;
    
    // thread main loop
    queued_query_t to_send[batch_max_size];
    auto wait_start = std::chrono::steady_clock::now();
    auto batch_time = std::chrono::microseconds(batch_time_us);
    uint64_t batch_id = 0;
    while(true){
        std::unique_lock<std::mutex> lock(thread_mtx);
        if(query_queue.empty()){
            thread_signal.wait_for(lock,batch_time);
        }

        if(!running) break;

        uint64_t send_count = 0;
        uint64_t queued_count = query_queue.size();
        auto now = std::chrono::steady_clock::now();

        if((queued_count >= batch_min_size) || ((now-wait_start) >= batch_time)){
            send_count = std::min(queued_count,batch_max_size);
            wait_start = now;

            // copy out queries
            for(uint64_t i=0;i<send_count;i++){
                to_send[i] = query_queue.front();
                query_queue.pop();
            }
        }

        lock.unlock();

        // now we are outside the locked region (i.e the client can continue adding queries to the queue): build object and call trigger_put
        if(send_count > 0){
            ObjectWithStringKey obj;
            obj.key = "/rag/emb/centroids_search/client" + std::to_string(node_id) + "/qb" + std::to_string(batch_id);

            // build Blob from queries in to_send
            std::string batch_string;
            build_batch_string(batch_string,to_send,send_count,emb_dim);
            obj.blob = Blob(reinterpret_cast<const uint8_t*>(batch_string.c_str()), batch_string.size());
            
            // trigger put
            capi.trigger_put(obj);
            
            batch_size[batch_id] = send_count; // for statistics
            batch_id++;
        }
    }
}

// notification thread methods

VortexCascadeClient::NotificationThread::NotificationThread(VortexCascadeClient* vortex){
    this->vortex = vortex;
}

void VortexCascadeClient::NotificationThread::push_result(const Blob& result){
    std::unique_lock<std::mutex> lock(thread_mtx);
    to_process.emplace(result);
    thread_signal.notify_all();
}

void VortexCascadeClient::NotificationThread::signal_stop(){
    std::unique_lock<std::mutex> lock(thread_mtx);
    running = false;
    thread_signal.notify_all();
}

void VortexCascadeClient::NotificationThread::main_loop(){
    if(!running) return;
    
    // thread main loop
    while(true){
        std::unique_lock<std::mutex> lock(thread_mtx);
        if(to_process.empty()){
            thread_signal.wait(lock);
        }

        if(!running) break;

        Blob blob(std::move(to_process.front()));
        to_process.pop();
        lock.unlock();
        
        // deserialize the JSON
        // TODO why JSON?
       
        if (blob.size == 0) {
             std::cerr << "Error: empty result blob." << std::endl;
             continue;
        }

        char* json_data = const_cast<char*>(reinterpret_cast<const char*>(blob.bytes));
        std::size_t json_size = blob.size;
        std::string json_string(json_data, json_size);

        try{
             nlohmann::json parsed_json = nlohmann::json::parse(json_string);
             vortex->result_received(parsed_json);
        } catch (const nlohmann::json::parse_error& e) {
             std::cerr << "Result JSON parse error: " << e.what() << std::endl;
             continue;
        }
    }
}


#include "benchmark_client.hpp"

VortexBenchmarkClient::VortexBenchmarkClient(){
}

VortexBenchmarkClient::~VortexBenchmarkClient(){
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

void VortexBenchmarkClient::setup(uint64_t batch_min_size,uint64_t batch_max_size,uint64_t batch_time_us,uint64_t emb_dim,uint64_t num_result_threads){
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

query_id_t VortexBenchmarkClient::next_query_id(){
    std::unique_lock<std::mutex> lock(query_id_mtx);
    query_id_t query_id = (static_cast<uint64_t>(my_id) << 48) | query_count;
    query_count++;
    return query_id;
}

uint64_t VortexBenchmarkClient::query(const std::string& query,const float* query_emb){
    query_id_t query_id = VortexBenchmarkClient::next_query_id();
    queued_query_t new_query(query_id,&query,query_emb);
    client_thread->push_query(new_query);
    return query_id;
}

void VortexBenchmarkClient::wait_results(){
    std::cout << "  received " << result_count << std::endl;
    uint64_t wait_time = 0;
    while(result_count < query_count){
        if(wait_time >= VORTEX_CLIENT_MAX_WAIT_TIME){
            std::cout << "  waited more than " << VORTEX_CLIENT_MAX_WAIT_TIME << " seconds, stopping ..." << std::endl;
            break;
        }

        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cout << "  received " << result_count << std::endl;
        wait_time += 2;
    }
}

void VortexBenchmarkClient::dump_timestamps(bool dump_remote){
    TimestampLogger::flush("client" + std::to_string(my_id) + ".dat");

    if(dump_remote){
        capi.dump_timestamp(UDL1_TIMESTAMP_FILE,UDL1_PATH);
        capi.dump_timestamp(UDL2_TIMESTAMP_FILE,UDL2_PATH);
        capi.dump_timestamp(UDL3_TIMESTAMP_FILE,UDL3_PATH);
    }
}

void VortexBenchmarkClient::result_received(nlohmann::json &all_results_json){
    for (const auto& result_json : all_results_json) {
        if (result_json.count("query") == 0 || result_json.count("top_k_docs") == 0) {
            std::cerr << "Result JSON does not contain query or top_k_docs." << std::endl;
            continue;
        }

        uint64_t batch_id = (int)result_json["query_batch_id"] / QUERY_BATCH_ID_MODULUS; // TODO this is weird: we should use a global unique identifier for each individual query
        std::string query_text(std::move(result_json["query"]));

        std::shared_lock<std::shared_mutex> lock(client_thread->map_mutex);
        auto index_and_id = client_thread->batched_query_to_index_and_id[batch_id][query_text];
        lock.unlock();

        uint64_t query_index = index_and_id.first;
        query_id_t query_id = index_and_id.second;

        std::unique_lock<std::shared_mutex> lock2(result_mutex);
        result.emplace(query_id,std::move(result_json["top_k_docs"]));
        lock2.unlock();
        
        TimestampLogger::log(LOG_TAG_QUERIES_RESULT_CLIENT_RECEIVED,my_id,batch_id,query_index);
        result_count++;
    }
}

const std::vector<std::string>& VortexBenchmarkClient::get_result(query_id_t query_id){
    std::shared_lock<std::shared_mutex> lock(result_mutex);
    return result[query_id];
}

// client thread methods

VortexBenchmarkClient::ClientThread::ClientThread(uint64_t batch_min_size,uint64_t batch_max_size,uint64_t batch_time_us,uint64_t emb_dim){
    this->batch_min_size = batch_min_size;
    this->batch_max_size = batch_max_size;
    this->batch_time_us = batch_time_us;
    this->emb_dim = emb_dim;
}

void VortexBenchmarkClient::ClientThread::push_query(queued_query_t &queued_query){
    std::unique_lock<std::mutex> lock(thread_mtx);
    query_queue.push(queued_query);
    thread_signal.notify_all();
}

void VortexBenchmarkClient::ClientThread::signal_stop(){
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
        query_list.push_back(*std::get<1>(to_send[i]));
    }

    batch_string += nlohmann::json(query_list).dump();
}

void VortexBenchmarkClient::ClientThread::main_loop(){
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
            obj.key = UDL1_PATH "/client" + std::to_string(node_id) + "/qb" + std::to_string(batch_id); // batch_id is important for randomizing the shard that gets each batch

            // build Blob from queries in to_send
            
            std::unordered_map<query_id_t,uint32_t> query_index; // maps query_id to position in the buffer
            uint32_t total_size = 0;
            uint32_t metadata_size = sizeof(uint32_t) * 5;
            uint32_t query_emb_size = sizeof(float) * emb_dim;

            // compute the number of bytes each query will take in the buffer
            for(uint64_t i=0;i<send_count;i++){
                query_id_t query_id = std::get<0>(to_send[i]);
                const std::string& query_txt = *std::get<1>(to_send[i]);
                const float* query_emb = std::get<2>(to_send[i]);

                uint32_t query_text_size = mutils::bytes_size(query_txt);
                total_size += query_text_size + metadata_size + query_emb_size;
                query_index[query_id] = query_text_size; // save this here temporarily
            }

            uint32_t index_size = mutils::bytes_size(query_index);
            total_size += index_size; // total buffer size

            // use a lambda to build buffer, to avoid a copy
            obj.blob = Blob([&](uint8_t* buffer,const std::size_t size){
                    uint32_t metadata_position = index_size; // position to start writing metadata
                    uint32_t embeddings_position = metadata_position + (send_count * metadata_size); // position to start writing the embeddings
                    uint32_t text_position = embeddings_position + (send_count * query_emb_size); // position to start writing the query texts

                    // write each query to the buffer, starting at buffer_position
                    for(uint64_t i=0;i<send_count;i++){
                        query_id_t query_id = std::get<0>(to_send[i]);
                        const std::string& query_txt = *std::get<1>(to_send[i]);
                        const float* query_emb = std::get<2>(to_send[i]);

                        uint32_t query_text_size = query_index[query_id];
                        query_index[query_id] = metadata_position; // update with the position where the metadata for this query is

                        // write metadata: node_id, query_text_position, query_text_size, embeddings_position, query_emb_size
                        uint32_t metadata_array[5] = {node_id,text_position,query_text_size,embeddings_position,query_emb_size};
                        std::memcpy(buffer+metadata_position,metadata_array,metadata_size);

                        // write embeddings
                        std::memcpy(buffer+embeddings_position,query_emb,query_emb_size);
                        
                        // write text
                        mutils::to_bytes(query_txt,buffer+text_position);
                       
                        // update position for the next 
                        metadata_position += metadata_size;
                        embeddings_position += query_emb_size;
                        text_position += query_text_size;
                    }

                    // write index
                    mutils::to_bytes(query_index,buffer);

                    return size;
                },total_size);

            for(uint64_t i=0;i<send_count;i++){
                auto query_id = std::get<0>(to_send[i]);
                TimestampLogger::log(LOG_TAG_QUERIES_SENDING_START,node_id,batch_id,i); // TODO update with just query_id
                
                std::unique_lock<std::shared_mutex> lock(map_mutex);
                batched_query_to_index_and_id[batch_id][*std::get<1>(to_send[i])] = std::make_pair(i,query_id); // TODO this will not be necessary
            }

            // trigger put
            capi.trigger_put(obj);
            
            for(uint64_t i=0;i<send_count;i++){
                TimestampLogger::log(LOG_TAG_QUERIES_SENDING_END,node_id,batch_id,i); // TODO update with just query_id
            }

            batch_size[batch_id] = send_count; // for statistics
            batch_id++;
        }
    }
}

// notification thread methods

VortexBenchmarkClient::NotificationThread::NotificationThread(VortexBenchmarkClient* vortex){
    this->vortex = vortex;
}

void VortexBenchmarkClient::NotificationThread::push_result(const Blob& result){
    std::unique_lock<std::mutex> lock(thread_mtx);
    to_process.emplace(result);
    thread_signal.notify_all();
}

void VortexBenchmarkClient::NotificationThread::signal_stop(){
    std::unique_lock<std::mutex> lock(thread_mtx);
    running = false;
    thread_signal.notify_all();
}

void VortexBenchmarkClient::NotificationThread::main_loop(){
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


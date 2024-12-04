#include <iostream>    
#include <limits>      
#include <stdexcept>   
#include <unordered_set>
#include <cascade/utils.hpp>
#include "serialize_utils.hpp"


/*
 * EmbeddingQueryBatcher implementation
 */

EmbeddingQueryBatcher::EmbeddingQueryBatcher(uint64_t emb_dim,uint64_t size_hint){
    this->emb_dim = emb_dim;
    metadata_size = sizeof(uint32_t) * 5 + sizeof(query_id_t);
    header_size = sizeof(uint32_t) * 2;
    query_emb_size = sizeof(float) * emb_dim;
    
    queries.reserve(size_hint);
    buffered_queries.reserve(size_hint);
}

void EmbeddingQueryBatcher::add_query(queued_query_t &queued_query){
    from_buffered = false;
    queries.emplace_back(queued_query);
}

void EmbeddingQueryBatcher::add_query(query_id_t query_id,uint32_t node_id,std::shared_ptr<float> query_emb,std::shared_ptr<std::string> query_text){
    queued_query_t queued_query(query_id,node_id,query_emb,query_text);
    add_query(queued_query);
}

void EmbeddingQueryBatcher::add_query(std::shared_ptr<EmbeddingQuery> query){
    from_buffered = true;
    buffered_queries.push_back(query);
}

void EmbeddingQueryBatcher::serialize(){
    if(from_buffered){
        serialize_from_buffered();
    } else {
        serialize_from_raw();
    }
}
    
void EmbeddingQueryBatcher::serialize_from_buffered(){
    num_queries = buffered_queries.size();
    total_text_size = 0;
    uint32_t total_size = header_size; // header: num_queries, embeddings_start_position

    // compute the number of bytes each query will take in the buffer
    for(auto& query : buffered_queries){
        query_id_t query_id = query->get_id();
        uint32_t query_text_size = query->get_text_size();
        total_size += query_text_size + metadata_size + query_emb_size;
        total_text_size += query_text_size;
        text_size[query_id] = query_text_size;
    }

    // use a lambda to build buffer, to avoid a copy
    blob = std::make_shared<derecho::cascade::Blob>([&](uint8_t* buffer,const std::size_t size){
            uint32_t metadata_position = header_size; // position to start writing metadata
            uint32_t text_position = metadata_position + (num_queries * metadata_size); // position to start writing the query texts
            uint32_t embeddings_position = text_position + total_text_size; // position to start writing the embeddings
            
            // write header
            uint32_t header[2] = {num_queries,embeddings_position};
            std::memcpy(buffer,header,header_size);
            
            // write each query to the buffer, starting at buffer_position
            for(auto& query : buffered_queries){
                query_id_t query_id = query->get_id();
                uint32_t node_id = query->get_node();
                const float* query_emb = query->get_embeddings_pointer();
                const uint8_t * text_data = query->get_text_pointer();
                uint32_t query_text_size = text_size[query_id];

                // write metadata: query_id, node_id, query_text_position, query_text_size, embeddings_position, query_emb_size
                uint32_t metadata_array[5] = {node_id,text_position,query_text_size,embeddings_position,query_emb_size};
                std::memcpy(buffer+metadata_position,&query_id,sizeof(query_id_t));
                std::memcpy(buffer+metadata_position+sizeof(query_id_t),metadata_array,metadata_size-sizeof(query_id_t));

                // write embeddings
                std::memcpy(buffer+embeddings_position,query_emb,query_emb_size);
                
                // write text
                std::memcpy(buffer+text_position,text_data,query_text_size);
               
                // update position for the next 
                metadata_position += metadata_size;
                embeddings_position += query_emb_size;
                text_position += query_text_size;
            }

            return size;
        },total_size);
}

void EmbeddingQueryBatcher::serialize_from_raw(){
    num_queries = queries.size();
    total_text_size = 0;
    uint32_t total_size = header_size; // header: num_queries, embeddings_start_position

    // compute the number of bytes each query will take in the buffer
    for(auto& queued_query : queries){
        query_id_t query_id = std::get<0>(queued_query);
        const std::string& query_txt = *std::get<3>(queued_query);

        uint32_t query_text_size = mutils::bytes_size(query_txt);
        total_text_size += query_text_size;
        total_size += query_text_size + metadata_size + query_emb_size;
        text_size[query_id] = query_text_size;
    }

    // use a lambda to build buffer, to avoid a copy
    blob = std::make_shared<derecho::cascade::Blob>([&](uint8_t* buffer,const std::size_t size){
            uint32_t metadata_position = header_size; // position to start writing metadata
            uint32_t text_position = metadata_position + (num_queries * metadata_size); // position to start writing the query texts
            uint32_t embeddings_position = text_position + total_text_size; // position to start writing the embeddings
             
            // write header
            uint32_t header[2] = {num_queries,embeddings_position};
            std::memcpy(buffer,header,header_size);

            // write each query to the buffer, starting at buffer_position
            for(auto& queued_query : queries){
                query_id_t query_id = std::get<0>(queued_query);
                uint32_t node_id = std::get<1>(queued_query);
                const float* query_emb = std::get<2>(queued_query).get();
                const std::string& query_txt = *std::get<3>(queued_query);
                uint32_t query_text_size = text_size[query_id];

                // write metadata: query_id, node_id, query_text_position, query_text_size, embeddings_position, query_emb_size
                uint32_t metadata_array[5] = {node_id,text_position,query_text_size,embeddings_position,query_emb_size};
                std::memcpy(buffer+metadata_position,&query_id,sizeof(query_id_t));
                std::memcpy(buffer+metadata_position+sizeof(query_id_t),metadata_array,metadata_size-sizeof(query_id_t));

                // write embeddings
                std::memcpy(buffer+embeddings_position,query_emb,query_emb_size);
                
                // write text
                mutils::to_bytes(query_txt,buffer+text_position);
               
                // update position for the next 
                metadata_position += metadata_size;
                embeddings_position += query_emb_size;
                text_position += query_text_size;
            }

            return size;
        },total_size);
}

std::shared_ptr<derecho::cascade::Blob> EmbeddingQueryBatcher::get_blob(){
    return blob;
}

void EmbeddingQueryBatcher::reset(){
    blob.reset();
    queries.clear();
    buffered_queries.clear();
    text_size.clear();
}

/*
 * EmbeddingQuery implementation
 */

EmbeddingQuery::EmbeddingQuery(std::shared_ptr<uint8_t> buffer,uint64_t buffer_size,uint64_t query_id,uint32_t metadata_position){
    this->buffer = buffer;
    this->buffer_size = buffer_size;
    this->query_id = query_id;

    // get metadata
    const uint32_t *metadata = reinterpret_cast<uint32_t*>(buffer.get()+metadata_position+sizeof(query_id_t));
    node_id = metadata[0];
    text_position = metadata[1];
    text_size = metadata[2];
    embeddings_position = metadata[3];
    embeddings_size = metadata[4];
}

std::shared_ptr<std::string> EmbeddingQuery::get_text(){
    if(!text){
        text = mutils::from_bytes<std::string>(nullptr,buffer.get()+text_position);
    }

    return text;
}

const float * EmbeddingQuery::get_embeddings_pointer(){
    if(embeddings_position >= buffer_size){
        return nullptr;
    }

    return reinterpret_cast<float*>(buffer.get()+embeddings_position);
}

const uint8_t * EmbeddingQuery::get_text_pointer(){
    return buffer.get()+text_position;
}

uint32_t EmbeddingQuery::get_text_size(){
    return text_size;
}

uint64_t EmbeddingQuery::get_id(){
    return query_id;
}

uint32_t EmbeddingQuery::get_node(){
    return node_id;
}


/*
 * EmbeddingQueryBatchManager implementation
 */

EmbeddingQueryBatchManager::EmbeddingQueryBatchManager(const uint8_t *buffer,uint64_t buffer_size,uint64_t emb_dim,bool copy_embeddings){
    this->emb_dim = emb_dim;
    this->copy_embeddings = copy_embeddings;
    
    const uint32_t *header = reinterpret_cast<const uint32_t *>(buffer);
    this->num_queries = header[0];
    this->embeddings_position = header[1];
    
    this->header_size = sizeof(uint32_t) * 2;
    this->metadata_size = sizeof(uint32_t) * 5 + sizeof(query_id_t);
    this->text_position = this->header_size + (this->metadata_size * this->num_queries);
    this->embeddings_size = buffer_size - this->embeddings_position;
   
    if(copy_embeddings){
        this->buffer_size = buffer_size;
    } else {
        this->buffer_size = buffer_size - this->embeddings_size;
    }

    std::shared_ptr<uint8_t> copy(new uint8_t[this->buffer_size]);
    std::memcpy(copy.get(),buffer,this->buffer_size);
    this->buffer = std::move(copy);
}

const std::vector<std::shared_ptr<EmbeddingQuery>>& EmbeddingQueryBatchManager::get_queries(){
    if(queries.empty()){
        create_queries();
    }

    return queries;
}

uint64_t EmbeddingQueryBatchManager::count(){
    return num_queries;
}

uint32_t EmbeddingQueryBatchManager::get_embeddings_position(uint32_t start){
    return embeddings_position + (start * (emb_dim * sizeof(float)));
}

uint32_t EmbeddingQueryBatchManager::get_embeddings_size(uint32_t num){
    if(num == 0){
        return this->embeddings_size;
    }

    return num * emb_dim * sizeof(float);
}

void EmbeddingQueryBatchManager::create_queries(){
    for(uint32_t i=0;i<num_queries;i++){
        uint32_t metadata_position = header_size + (i * metadata_size);
        query_id_t query_id = *reinterpret_cast<query_id_t*>(buffer.get()+metadata_position);
        queries.emplace_back(new EmbeddingQuery(buffer,buffer_size,query_id,metadata_position));
    }
}

/*
 * Helper functions
 *
 */

std::pair<uint32_t,uint64_t> parse_client_and_batch_id(const std::string &str){
    size_t pos = str.find("_");
    uint32_t client_id = std::stoll(str.substr(0,pos));
    uint64_t batch_id = std::stoull(str.substr(pos+1));
    return std::make_pair(client_id,batch_id);
}

uint64_t parse_cluster_id(const std::string &str){
    return std::stoull(str.substr(8)); // str is '/cluster[0-9]+'
}

// old stuff below ========================================

std::string format_query_emb_object(int nq, std::unique_ptr<float[]>& xq, std::vector<std::string>& query_list, uint32_t embedding_dim) {
     // create an bytes object by concatenating: num_queries + float array of emebddings + list of query_text
     uint32_t num_queries = static_cast<uint32_t>(nq);
     std::string nq_bytes(4, '\0');
     nq_bytes[0] = (num_queries >> 24) & 0xFF;
     nq_bytes[1] = (num_queries >> 16) & 0xFF;
     nq_bytes[2] = (num_queries >> 8) & 0xFF;
     nq_bytes[3] = num_queries & 0xFF;
     float* query_embeddings = xq.get();
     // serialize the query embeddings and query texts, formated as num_queries + query_embeddings + query_texts
     std::string query_emb_string = nq_bytes +
                              std::string(reinterpret_cast<const char*>(query_embeddings), sizeof(float) * embedding_dim * num_queries) +
                              nlohmann::json(query_list).dump();
     return query_emb_string;
}

/***
* Helper function for logging purpose, to extract the query information from the key
* @param key_string the key string to extract the query information from
* @param delimiter the delimiter to separate the number from the key string
* @param number the number extracted from the key string
* @note the function truncate the number string if it exceeds the length that an int can handle
***/
bool parse_number(const std::string& key_string, const std::string& delimiter, int& number) {
     size_t pos = key_string.find(delimiter);
     if (pos == std::string::npos) {
          return false;
     }
     pos += delimiter.size();
     std::string number_str;
     while (pos < key_string.size() && std::isdigit(key_string[pos])) {
          number_str += key_string[pos];
          ++pos;
     }
     if (number_str.empty()) {
          return false;
     }
     /*** Truncate the number string to fit into an int if necessary
     * down by 2 digits to make sure it doesn't out of range    
     * TODO: better way to do this?     
     */
     constexpr size_t safe_digits = std::numeric_limits<int>::digits10 - 2; 
     if (number_str.length() > safe_digits) {
          number_str = number_str.substr(0, safe_digits); 
     }
     try {
          number = std::stoi(number_str);  // Convert the truncated string to an int
     } catch (const std::invalid_argument& e) {
          std::cerr << "Failed to parse number from key: " << key_string << std::endl;
          return false;
     } 
     return true;
}


bool parse_batch_id(const std::string& key_string, int& client_id, int& batch_id) {
     if (!parse_number(key_string, "client", client_id)) {
          std::cerr << "Failed to parse client_id from key: " << key_string << std::endl;
          return false;
     }
     if (!parse_number(key_string, "qb", batch_id)) {
          std::cerr << "Failed to parse batch_id from key: " << key_string << std::endl;
          return false;
     }
     return true;
}


bool parse_query_info(const std::string& key_string, int& client_id, int& batch_id, int& cluster_id, int& qid){
     if (!parse_number(key_string, "client", client_id)) {
          std::cerr << "Failed to parse client_id from key: " << key_string << std::endl;
          return false;
     }
     if (!parse_number(key_string, "qb", batch_id)) {
          std::cerr << "Failed to parse batch_id from key: " << key_string << std::endl;
          return false;
     }
     if (!parse_number(key_string, "_cluster", cluster_id)) {
          std::cerr << "Failed to parse cluster_id from key: " << key_string << std::endl;
          return false;
     }
     if (!parse_number(key_string, "_qid", qid)) {
          std::cerr << "Failed to parse qid from key: " << key_string << std::endl;
          return false;
     }
     return true;
}

/*** Helper function to callers of list_key:
*    filter keys that doesn't have exact prefix, or duplicate keys (from experiment at scale, it occurs.)
*    e.g. /doc1/1, /doc12/1, which return by list_keys("/doc1"), but are not for the same cluster
*    TODO: adjust op_list_keys semantics? 
*/
std::priority_queue<std::string, std::vector<std::string>, CompareObjKey> filter_exact_matched_keys(std::vector<std::string>& obj_keys, const std::string& prefix){
     std::priority_queue<std::string, std::vector<std::string>, CompareObjKey> filtered_keys;
     std::unordered_set<std::string> key_set; /*** TODO: only for correctness test*/
     for (auto& key : obj_keys) {
          size_t pos = key.rfind("/");
          if (pos == std::string::npos) {
               std::cerr << "Error: invalid obj_key format, key=" << key << "prefix" << prefix  << std::endl; // shouldn't happen
               continue;
          }
          if (key.substr(0, pos) == prefix && key_set.find(key) == key_set.end()) {
               filtered_keys.push(key);
               key_set.insert(key);
          }
     }
     if (key_set.size() != filtered_keys.size()) {
          std::cerr << "Error: filter_exact_matched_keys: key_set.size()=" << key_set.size() << ",filtered_keys.size()=" << filtered_keys.size() << std::endl;
     }
     return filtered_keys;
}

/*** 
* Helper function to cdpo_handler()
* @param bytes the bytes object to deserialize
* @param data_size the size of the bytes object
* @param nq the number of queries in the blob object, output. Used by FAISS search.
     type is uint32_t because in previous encode_centroids_search_udl, it is serialized from an unsigned "big" ordered int
* @param query_embeddings the embeddings of the queries, output. 
***/
void deserialize_embeddings_and_quries_from_bytes(const uint8_t* bytes,
                                                            const std::size_t& data_size,
                                                            uint32_t& nq,
                                                            const int& emb_dim,
                                                            float*& query_embeddings,
                                                            std::vector<std::string>& query_list) {
     if (data_size < 4) {
          throw std::runtime_error("Data size is too small to deserialize its embeddings and queries.");
     }
     
     // 0. get the number of queries in the blob object
     nq = (static_cast<uint32_t>(bytes[0]) << 24) |
                    (static_cast<uint32_t>(bytes[1]) << 16) |
                    (static_cast<uint32_t>(bytes[2]) <<  8) |
                    (static_cast<uint32_t>(bytes[3]));
     dbg_default_trace("In [{}],Number of queries: {}",__func__,nq);
     // 1. get the emebddings of the queries from the blob object
     std::size_t float_array_start = 4;
     std::size_t float_array_size = sizeof(float) * emb_dim * nq;
     std::size_t float_array_end = float_array_start + float_array_size;
     if (data_size < float_array_end) {
          std::cerr << "Data size "<< data_size <<" is too small for the expected float array end: " << float_array_end <<"." << std::endl;
          return;
     }
     query_embeddings = const_cast<float*>(reinterpret_cast<const float*>(bytes + float_array_start));

     // 2. get the queries from the blob object
     std::size_t json_start = float_array_end;
     if (json_start >= data_size) {
          std::cerr << "No space left for queries data." << std::endl;
          return;
     }
     // Create a JSON string from the remainder of the bytes object
     char* json_data = const_cast<char*>(reinterpret_cast<const char*>(bytes + json_start));
     std::size_t json_size = data_size - json_start;
     std::string json_string(json_data, json_size);

     // Parse the JSON string using nlohmann/json
     try {
          nlohmann::json parsed_json = nlohmann::json::parse(json_string);
          query_list = parsed_json.get<std::vector<std::string>>();
     } catch (const nlohmann::json::parse_error& e) {
          std::cerr << "JSON parse error: " << e.what() << std::endl;
     }
}

void new_deserialize_embeddings_and_quries_from_bytes(const uint8_t* bytes,
                                                            const std::size_t& data_size,
                                                            uint32_t& nq,
                                                            const int& emb_dim,
                                                            float*& query_embeddings,
                                                            std::vector<std::string>& query_list) {
    if (data_size < 4) {
        throw std::runtime_error("Data size is too small to deserialize its embeddings and queries.");
    }

    // index
    std::unique_ptr<std::unordered_map<uint64_t,uint32_t>> batch_index = mutils::from_bytes<std::unordered_map<uint64_t,uint32_t>>(nullptr,bytes);

    // XXX for compatibility, get the order in which queries are written (embeddings and text)
    // TODO this will not be needed after refactoring all UDLs
    std::vector<uint64_t> id_list;
    for(auto& item : *batch_index){
        id_list.push_back(item.first);
    }
    std::sort(id_list.begin(),id_list.end(),[&](const uint64_t& l, const uint64_t& r){
            return batch_index->at(l) < batch_index->at(r);
        });

    // 0. get the number of queries in the blob object
    nq = batch_index->size();
    dbg_default_trace("In [{}],Number of queries: {}",__func__,nq);

    // 1. get the embeddings of the queries from the blob object
    const uint32_t *metadata = reinterpret_cast<const uint32_t *>(bytes + batch_index->at(id_list.front()));
    uint32_t embeddings_position = metadata[3];
    query_embeddings = const_cast<float*>(reinterpret_cast<const float*>(bytes + embeddings_position));

    // 2. get the queries from the blob object
    for(auto query_id : id_list){
        metadata = reinterpret_cast<const uint32_t *>(bytes + batch_index->at(query_id));
        auto query_txt = *mutils::from_bytes<std::string>(nullptr,bytes + metadata[1]);
        query_list.emplace_back(std::move(query_txt));
    }
}

/***
* Format the search results for each query to send to the next UDL.
* The format is | top_k | embeding_id_vector | distance_vector | query_text |
***/
std::string serialize_cluster_search_result(uint32_t top_k, long* I, float* D, int idx, const std::string& query_text){
     std::string query_search_result;
     std::string num_embs(4, '\0');  // denotes the number of embedding_ids and distances 
     num_embs[0] = (top_k >> 24) & 0xFF;
     num_embs[1] = (top_k >> 16) & 0xFF;
     num_embs[2] = (top_k >> 8) & 0xFF;
     num_embs[3] = top_k & 0xFF;
     query_search_result = num_embs +\
                         std::string(reinterpret_cast<const char*>(&I[idx * top_k]) , sizeof(long) * top_k) +\
                         std::string(reinterpret_cast<const char*>(&D[idx * top_k]) , sizeof(float) * top_k) +\
                         query_text;
     return query_search_result; // RVO
}


/***
 * Helper function to aggregate cdpo_handler()
 * 
***/
void deserialize_cluster_search_result_from_bytes(const int& cluster_id,
                                                  const uint8_t* bytes,
                                                  const size_t& data_size,
                                                  std::string& query_text,
                                                  std::vector<DocIndex>& cluster_results) {
     if (data_size < 4) {
          throw std::runtime_error("Data size is too small to deserialize the cluster searched result.");
     }
     
     // 0. get the count of top_k selected from this cluster in the blob object
     uint32_t cluster_selected_count = (static_cast<uint32_t>(bytes[0]) << 24) |
                    (static_cast<uint32_t>(bytes[1]) << 16) |
                    (static_cast<uint32_t>(bytes[2]) <<  8) |
                    (static_cast<uint32_t>(bytes[3]));
     dbg_default_trace("In [{}], cluster searched top_k: {}",__func__,cluster_selected_count);
     // 1. get the cluster searched top_k emb index vector (I) from the blob object
     std::size_t I_array_start = 4;
     std::size_t I_array_size = sizeof(long) * cluster_selected_count;
     std::size_t I_array_end = I_array_start + I_array_size;
     if (data_size < I_array_end) {
          std::cerr << "Data size "<< data_size <<" is too small for the I array: " << I_array_end <<"." << std::endl;
          return;
     }
     long* I = const_cast<long*>(reinterpret_cast<const long*>(bytes + I_array_start));
     // 2. get the distance vector (D)
     std::size_t D_array_start = I_array_end;
     std::size_t D_array_size = sizeof(float) * cluster_selected_count;
     std::size_t D_array_end = D_array_start + D_array_size;
     if (data_size < D_array_end) {
          std::cerr << "Data size "<< data_size <<" is too small for the D array: " << D_array_end <<"." << std::endl;
          return;
     }
     float* D = const_cast<float*>(reinterpret_cast<const float*>(bytes + D_array_start));
     // 3. get the query text
     std::size_t query_text_start = D_array_end;
     if (query_text_start >= data_size) {
          std::cerr << "No space left for query text." << std::endl;
          return;
     }
     // convert the remaining bytes to std::string
     char* query_text_data = const_cast<char*>(reinterpret_cast<const char*>(bytes + query_text_start));
     std::size_t query_text_size = data_size - query_text_start;
     query_text = std::string(query_text_data, query_text_size);
     // 4. create the DocIndex vector
     for (uint32_t i = 0; i < cluster_selected_count; i++) {
          cluster_results.push_back(DocIndex{cluster_id, I[i], D[i]});
     }

}

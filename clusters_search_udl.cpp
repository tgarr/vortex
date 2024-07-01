#include <cascade/user_defined_logic_interface.hpp>
#include <cascade/cascade_interface.hpp>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <map>

#include "grouped_embeddings.hpp"


namespace derecho{
namespace cascade{

#define MY_UUID     "11a2c123-2200-21ac-1755-0002ac220000"
#define MY_DESC     "UDL search within the clusters to find the top K embeddings that the queries close to."

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

class ClustersSearchOCDPO: public DefaultOffCriticalDataPathObserver {

    // maps from cluster ID -> embeddings of that cluster, 
    // because there could be more than 1 clusters hashed to one node by affinity set.
    std::unordered_map<int, std::unique_ptr<GroupedEmbeddings>> clusters_embs;

    // faiss example uses 64. These two values could be set by config in dfgs.json.tmp file
    int emb_dim = 64; // dimension of each embedding
    int top_k = 4; // number of top K embeddings to search

    /*** 
     * Helper function to fill_cluster_embs_in_cache()
     * Retrieve the embeddings of a single object from the KV store in Cascade. 
     * This retrive doesn't involve copying the data.
     * @param cluster_num_embs the number of embeddings in the cluster
     * @param cluster_emb_key the key of the object to retrieve
     * @param typed_ctxt the context to get the service client reference
     * @param version the version of the object to retrieve
     * @param stable whether to get the stable version of the object
    ***/
    float* single_emb_object_retrieve(int& cluster_num_embs,
                                        std::string& cluster_emb_key,
                                        DefaultCascadeContextType* typed_ctxt,
                                        persistent::version_t version,
                                        bool stable = 1){
        float* data;
        // 1. get the object from KV store
        auto get_query_results = typed_ctxt->get_service_client_ref().get(cluster_emb_key,version, stable);
        auto& reply = get_query_results.get().begin()->second.get();
        Blob blob = std::move(const_cast<Blob&>(reply.blob));
        blob.memory_mode = derecho::cascade::object_memory_mode_t::EMPLACED; // Avoid copy, use bytes from reply.blob, transfer its ownership to GroupedEmbeddings.emb_data
        // 2. get the embeddings from the object
        data = const_cast<float*>(reinterpret_cast<const float *>(blob.bytes));
        size_t num_points = blob.size / sizeof(float);
        cluster_num_embs += num_points / this->emb_dim;
        return data;
    }

    /*** 
     * Helper function to fill_cluster_embs_in_cache()
     * Retrieve the embeddings of multiple objects from the KV store in Cascade. 
     * This involve copying the data from received blobs.
    ***/
    float* multi_emb_object_retrieve(int& cluster_num_embs,
                                    std::vector<std::string>& cluster_emb_obj_keys,
                                    DefaultCascadeContextType* typed_ctxt,
                                    persistent::version_t version,
                                    bool stable = 1){
        float* data;
        size_t num_obj = cluster_emb_obj_keys.size();
        size_t data_size = 0;
        Blob blobs[num_obj];
        for (size_t i = 0; i < num_obj; i++) {
            auto get_query_results = typed_ctxt->get_service_client_ref().get(cluster_emb_obj_keys[i],version, stable);
            auto& reply = get_query_results.get().begin()->second.get();
            blobs[i] = std::move(const_cast<Blob&>(reply.blob));
            data_size += blobs[i].size / sizeof(float);
        }
        // 2. copy the embeddings from the blobs to the data
        data = (float*)malloc(data_size * sizeof(float));
        size_t offset = 0;
        for (size_t i = 0; i < num_obj; i++) {
            memcpy(data + offset, blobs[i].bytes, blobs[i].size);
            offset += blobs[i].size / sizeof(float);
        }
        cluster_num_embs = data_size / this->emb_dim;
        return data;
    }

    /***
     * Fill in the memory cache by getting the clusters' embeddings from KV store in Cascade
     * This function is called when this UDL is first triggered by caller to operator(),
     * it sacrifices the first request to this node, but the following requests will benefit from this cache.
     * In static RAG setting, this function should be called only once at the begining
     * In dynamic RAG setting, this function could be extended to call periodically or upon notification 
     * (The reason of not filling it at initialization, is that initialization is called upon server starts, 
     *  but the data have not been put to the servers yet, this needs to be called after the clusters' embeddings data are put)
     * @param cluster_id the cluster ID to get the embeddings
     * @param typed_ctxt the context to get the service client reference
     * @return 0 on success, -1 on failure
     * @note we load the stabled version of the cluster embeddings
    ***/
    int fill_cluster_embs_in_cache(int cluster_id, 
                                    DefaultCascadeContextType* typed_ctxt){
        bool stable = 1; 
        persistent::version_t version = CURRENT_VERSION;
        // 0. check the keys for this cluster embedding objects stored in cascade
        //    because of the message size, one cluster might need multiple objects to store its embeddings
        std::string cluster_prefix = "/rag/emb/cluster" + std::to_string(cluster_id); /*** TODO: move it to config, the /rag/emb/cluster is hard-coded here ***/
        auto keys_future = typed_ctxt->get_service_client_ref().list_keys(version, stable, cluster_prefix);
        std::vector<std::string> cluster_emb_obj_keys = typed_ctxt->get_service_client_ref().wait_list_keys(keys_future);
        if (cluster_emb_obj_keys.empty()) {
            std::cerr << "Error: cluster" << cluster_id <<" has no cluster embeddings found in the KV store" << std::endl;
            dbg_default_error("Failed to find cluster embeddings in the KV store, at clusters_search_udl.");
            return -1;
        }

        // 1. Get the cluster embeddings from KV store in Cascade
        float* data;
        int cluster_num_embs = 0;
        if (cluster_emb_obj_keys.size() == 1) {
            data = single_emb_object_retrieve(cluster_num_embs, cluster_emb_obj_keys[0], typed_ctxt, version, stable);
        } else {
            data = multi_emb_object_retrieve(cluster_num_embs, cluster_emb_obj_keys, typed_ctxt, version ,stable);
        }
        if (cluster_num_embs == 0) {
            std::cerr << "Error: cluster" << cluster_id <<" has no embeddings found in the KV store" << std::endl;
            dbg_default_error("Failed to find cluster embeddings in the KV store, at clusters_search_udl.");
            return -1;
        }

        std::cout << "[ClustersSearchOCDPO]: num_emb_objects=" << cluster_num_embs << std::endl;
        
        // 2. fill in the memory cache
        this->clusters_embs[cluster_id]= std::make_unique<GroupedEmbeddings>(this->emb_dim, cluster_num_embs, data);
        this->clusters_embs[cluster_id]->initialize_cpu_flat_search(); // use CPU search in ocdpo search
        return 0;
    }

    /***
     * Get the cluster ID from the object's key_string
     * This function is called when the object is received from the sender
     * The cluster ID is used to get the embeddings of the corresponding cluster
     * @param key_string the key string of the object
     * @return the cluster ID, -1 if not found
    ***/
    inline int get_cluster_id(const std::string& key_string) {
        size_t pos = key_string.find("cluster");
        if (pos == std::string::npos) {
            return -1;
        }
        pos += 7; 
        // Extract the number following "cluster"
        std::string numberStr;
        while (pos < key_string.size() && std::isdigit(key_string[pos])) {
            numberStr += key_string[pos];
            ++pos;
        }
        if (!numberStr.empty()) {
            return std::stoi(numberStr);
        }
        return -1;
    }

    /***
     * Format the new_keys for the search results of the queries
     * it is formated as client{client_id}qb{querybatch_id}qc{client_batch_query_count}_cluster{cluster_id}_{hash(query)}
     * @param new_keys the new keys to be constructed
     * @param key_string the key string of the object
     * @param query_dict {query_id: query_string}
    ***/
    inline void construct_new_keys(std::vector<std::string>& new_keys,
                                                       const std::string& key_string, 
                                                       const std::map<int, std::string>& query_dict) {
        for (const auto& pair : query_dict) {
            std::string new_key = key_string + "_qid" + std::to_string(pair.first);
            new_keys.push_back(new_key);
        }
    }

    /*** 
     * Helper function to cdpo_handler()
    ***/
    void deserialize_embeddings_and_quries_from_bytes(const uint8_t* bytes,
                                                                const std::size_t data_size,
                                                                int32_t& nq,
                                                                float*& query_embeddings,
                                                                std::map<int, std::string>& query_dict,
                                                                std::vector<std::string>& queries) {
        
        // 0. get the number of queries in the blob object
        // TODO: direct cast
        nq =(static_cast<int32_t>(bytes[0]) << 24) |
                (static_cast<int32_t>(bytes[1]) << 16) |
                (static_cast<int32_t>(bytes[2]) << 8)  |
                (static_cast<int32_t>(bytes[3]));
        std::cout << "Number of queries: " << nq << std::endl;
        // 1. get the emebddings of the queries from the blob object
        std::size_t float_array_start = 4;
        std::size_t float_array_size = sizeof(float) * this->emb_dim * nq;
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
        nlohmann::json parsed_json;
        try {
            parsed_json = nlohmann::json::parse(json_string);
            // Note the map must be ordered according to the order of queries, 
            // which is how the json is constructed by encode_centroids_search_udl
            for (json::iterator it = parsed_json.begin(); it != parsed_json.end(); ++it) {
                int key = std::stoi(it.key());  // Convert key from string to int
                std::string value = it.value(); // Get the value (already a string)
                query_dict[key] = value;
                queries.emplace_back(value);
            }
        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
        }
    }

    virtual void ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) override {
        /*** TODO: this object_pool_pathname is trigger prefix: /rag/emb/clusteres_search instead of /rag/emb, i.e. the objp name***/
        std::cout << "[Clusters search ocdpo]: I(" << worker_id << ") received an object from sender:" << sender << " with key=" << key_string  << std::endl;
        // 0. get the cluster ID
        int cluster_id = get_cluster_id(key_string); // TODO: get the cluster ID from the object
        if (cluster_id == -1) {
            std::cerr << "Error: cluster ID not found in the key_string" << std::endl;
            dbg_default_error("Failed to find cluster ID from key: {}, at clusters_search_udl.", key_string);
            return;
        }

        // 1. compute knn for the corresponding cluster on this node
        // 1.1. check if local cache contains the embeddings of the cluster
        auto it = this->clusters_embs.find(cluster_id);
        if (it == this->clusters_embs.end()){
            int filled_cluster_embs = fill_cluster_embs_in_cache(cluster_id, typed_ctxt);
            if (filled_cluster_embs == -1) {
                std::cerr << "Error: failed to fill the cluster embeddings in cache" << std::endl;
                dbg_default_error("Failed to fill the cluster embeddings in cache, at clusters_search_udl.");
                return;
            }
            it = this->clusters_embs.find(cluster_id);
        }
        // 1.2. get the embeddings of the cluster
        auto& embs = it->second;

        // 2. get the query embeddings from the object

        float* data;
        int32_t nq;
        std::map<int, std::string> query_dict;
        std::vector<std::string> queries;
        deserialize_embeddings_and_quries_from_bytes(object.blob.bytes,object.blob.size,nq,data,query_dict, queries);

        // 3. search the top K embeddings that are close to the query
        long* I = new long[this->top_k * nq];
        float* D = new float[this->top_k * nq];
        embs->faiss_cpu_flat_search(nq, data, this->top_k, D, I);

        // 4. emit the result to the subsequent UDL
        // 4.1 construct new keys for all queries in this search
        std::vector<std::string> new_keys;
        construct_new_keys(new_keys, key_string, query_dict);
        for (int idx = 0; idx < nq; idx++) {
            // 4.2 construct the cluster search result of query idx
            nlohmann::json json_obj; // format it as {"emb_id1": distance1, ...}
            for (int j = 0; j < this->top_k; j++) {
                json_obj[std::to_string(I[idx * this->top_k + j])] = std::to_string(cluster_id) + "-" + std::to_string(D[idx * this->top_k + j]);
            }
            json_obj["query"] = queries[idx];
            std::string json_str = json_obj.dump();
            Blob blob(reinterpret_cast<const uint8_t*>(json_str.c_str()), json_str.size());
            // 4.3 emit the result
            emit(new_keys[idx], EMIT_NO_VERSION_AND_TIMESTAMP , blob);
            std::cout << "Emitted key: " << new_keys[idx] << ", blob: " << json_str << std::endl;
        }
        delete[] I;
        delete[] D;
        std::cout << "[Cluster search ocdpo]: FINISHED knn search for key: " << key_string << std::endl;
    }

    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;
public:

    static void initialize() {
        if(!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<ClustersSearchOCDPO>();
        }
    }
    static auto get() {
        return ocdpo_ptr;
    }

    void set_config(const nlohmann::json& config){
        try{
            if (config.contains("emb_dim")) {
                this->emb_dim = config["emb_dim"].get<int>();
            }
            if (config.contains("top_k")) {
                this->top_k = config["top_k"].get<int>();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to convert emb_dim or top_k from config" << std::endl;
            dbg_default_error("Failed to convert emb_dim or top_k from config, at clusters_search_udl.");
        }
    }
};

std::shared_ptr<OffCriticalDataPathObserver> ClustersSearchOCDPO::ocdpo_ptr;

void initialize(ICascadeContext* ctxt) {
    ClustersSearchOCDPO::initialize();
}

std::shared_ptr<OffCriticalDataPathObserver> get_observer(
        ICascadeContext*,const nlohmann::json& config) {
    std::static_pointer_cast<ClustersSearchOCDPO>(ClustersSearchOCDPO::get())->set_config(config);
    return ClustersSearchOCDPO::get();
}

void release(ICascadeContext* ctxt) {
    // nothing to release
    return;
}

} // namespace cascade
} // namespace derecho

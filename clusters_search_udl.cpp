#include <cascade/user_defined_logic_interface.hpp>
#include <cascade/cascade_interface.hpp>
#include <iostream>
#include <unordered_map>
#include <memory>

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
     * Fill in the memory cache by getting the clusters' embeddings from KV store in Cascade
     * This function is called when this UDL is first triggered by caller to operator(),
     * it sacrifices the first request to this node, but the following requests will benefit from this cache.
     * In static RAG setting, this function should be called only once at the begining
     * In dynamic RAG setting, this function could be extended to call periodically or upon notification 
     * (The reason of not filling it at initialization, is that initialization is called upon server starts, 
     *  but the data have not been put to the servers yet, this needs to be called after the clusters' embeddings data are put)
    ***/
    void fill_cluster_embs_in_cache(int cluster_id, 
                                    DefaultCascadeContextType* typed_ctxt){
        // 1. Get the cluster embeddings from KV store in Cascade
        // 1.0. construct the keys to get the cluster embeddings
        /*** TODO: use list key to get all cluster emb objects of this cluster
                   if it is more than one objects, copy objects to a memory for storing the concatenated embeddings.
                   otherwise, use emplace to avoid copy  ***/
        std::string cluster_emb_key = "/rag/emb/cluster" + std::to_string(cluster_id);
        // 1.1. get the object from KV store
        Blob blob;
        int cluster_num_embs = 0;
        bool stable = 1; // TODO: double check to see if this is the most efficient way
        persistent::version_t version = CURRENT_VERSION;
        auto result = typed_ctxt->get_service_client_ref().get(cluster_emb_key,version, stable);
        for(auto& reply_future : result.get()) {
            auto reply = reply_future.second.get();
            blob = reply.blob;
            /*** TODO: Figure out how is Blob object constructed and deconstructed for memory safety, 
                        check if this is correct way to get without additional copy ***/
            blob.memory_mode = derecho::cascade::object_memory_mode_t::EMPLACED;
            break;
        }
        // 1.2. get the embeddings from the object
        float* data = const_cast<float*>(reinterpret_cast<const float *>(blob.bytes));
        float (*emb_array)[this->emb_dim] = reinterpret_cast<float (*)[this->emb_dim]>(data); // convert 1D float array to 2D
        float* emb_data = &emb_array[0][0];
        size_t num_points = blob.size / sizeof(float);
        cluster_num_embs += num_points / this->emb_dim;
        std::cout << "[ClustersSearchOCDPO]: num_points=" << num_points << std::endl;
        // float* xb = new float[this->emb_dim * this->num_embs]; // Placeholder embeddings
        // 2. fill in the memory cache
        this->clusters_embs[cluster_id]= std::make_unique<GroupedEmbeddings>(this->emb_dim, cluster_num_embs, emb_data);
        this->clusters_embs[cluster_id]->initialize_cpu_flat_search(); // use CPU search in ocdpo search
        std::cout << "[ClustersSearchOCDPO] added Cluster[" << cluster_id << "] to cache"<< std::endl;
        return;
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
     * Get the qids from the key_string
     * The key string is formated as "client{client_id}qb{querybatch_id}{qids-topK}_cluster{cluster_id}",
     *     where qids are the query IDs separated by '-'.
     * Format it to "client{client_id}qb{querybatch_id}qid{qid}{topK}_cluster{cluster_id}" for each qids
     * @param key_string the key string of the object
    ***/
    inline std::vector<std::string> construct_new_keys(const std::string& key_string) {
        std::vector<std::string> new_keys;
        size_t prefix_pos = key_string.find("qids");
        size_t suffix_pos = key_string.find("top");
        if (prefix_pos == std::string::npos || suffix_pos == std::string::npos) {
            std::cerr << "Error: qids or top not found in the key_string" << std::endl;
            dbg_default_error("Failed to find qids or top from key: {}, at clusters_search_udl.", key_string);
            return new_keys;
        }
        std::string new_key_prefix = key_string.substr(0, prefix_pos) + "qid";
        std::string new_key_suffix = key_string.substr(suffix_pos);

        size_t pos = key_string.find("qids");
        if (pos == std::string::npos) {
            return new_keys;
        }
        pos += 4;  // Skip past "qids"
        std::string qids_str;
        while (pos < key_string.size() && (std::isdigit(key_string[pos]) || key_string[pos] == '-')) {
            qids_str += key_string[pos];
            ++pos;
        }
        std::istringstream qids_stream(qids_str);
        std::string qid;
        while (std::getline(qids_stream, qid, '-')) {
            if (!qid.empty()) {
                new_keys.push_back(new_key_prefix + qid + new_key_suffix);
            }
        }
        return new_keys; 
    }

    virtual void ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) override {
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
            fill_cluster_embs_in_cache(cluster_id, typed_ctxt);
            it = this->clusters_embs.find(cluster_id);
        }
        // 1.2. get the embeddings of the cluster
        auto& embs = it->second;

        // 2. get the query embeddings from the object
        float* data = const_cast<float*>(reinterpret_cast<const float *>(object.blob.bytes)); /*** TODO: check if need to delete this? ***/
        int nq = static_cast<int>(object.blob.size / sizeof(float)) / this->emb_dim;
        float (*emb_array)[this->emb_dim] = reinterpret_cast<float (*)[this->emb_dim]>(data); // convert 1D float array to 2D
        float* query_emb_data = &emb_array[0][0];

        // 3. search the top K embeddings that are close to the query
        long* I = new long[this->top_k * nq];
        float* D = new float[this->top_k * nq];
        embs->faiss_cpu_flat_search(nq, query_emb_data, this->top_k, D, I);

        // 4. emit the result to the subsequent UDL
        // 4.1 construct new keys for all queries in this search
        std::vector<std::string> new_keys = construct_new_keys(key_string);
        if (new_keys.empty() || new_keys.size() < static_cast<size_t>(nq)) {
            std::cerr << "Error: failed to construct new keys" << std::endl;
            dbg_default_error("Failed to construct new keys from key: {}, at clusters_search_udl.", key_string);
            return;
        }
        for (int idx = 0; idx < nq; idx++) {
            // 4.2 construct the cluster search result of query idx
            nlohmann::json json_obj; // format it as {"emb_id1": distance1, ...}
            for (int j = 0; j < this->top_k; j++) {
                json_obj[std::to_string(I[idx * this->top_k + j])] = D[idx * this->top_k + j];
            }
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

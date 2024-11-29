#pragma once
#include <queue>
#include <vector>
#include <string>
#include <cascade/service_client_api.hpp>

#define QUERY_BATCH_ID_MODULUS 100000
#define CLUSTER_KEY_DELIMITER "_cluster"

/* 
 * VortexEmbeddingQueryBatcher gathers and serializes queries with their embeddings to be sent to UDL1 and to UDL2.
 *
 */

using query_id_t = uint64_t;
using queued_query_t = std::tuple<query_id_t,uint32_t,std::shared_ptr<float>,std::shared_ptr<std::string>>; // query ID, client node ID, embeddings, query text

class VortexEmbeddingQueryBatcher {
    uint64_t emb_dim;
    uint32_t metadata_size;
    uint32_t query_emb_size;

    std::vector<queued_query_t> queries;
    std::shared_ptr<derecho::cascade::Blob> blob;
    std::unordered_map<query_id_t,uint32_t> query_index;

public:
    VortexEmbeddingQueryBatcher(uint64_t emb_dim,uint64_t size_hint = 1000);

    void add_query(queued_query_t &queued_query);
    void add_query(query_id_t query_id,uint32_t node_id,std::shared_ptr<float> query_emb,std::shared_ptr<std::string> query_text);

    uint64_t size();
    const std::vector<queued_query_t>& get_queries();
    std::shared_ptr<derecho::cascade::Blob> get_blob();

    void serialize();
    void reset();
};


/*
 * Helper functions
 *
 */

std::string format_query_emb_object(int nq, std::unique_ptr<float[]>& xq, std::vector<std::string>& query_list, uint32_t embedding_dim);

/***
* Helper function for logging purpose, to extract the query information from the key
* @param key_string the key string to extract the query information from
* @param delimiter the delimiter to separate the number from the key string
* @param number the number extracted from the key string
* @note the function truncate the number string if it exceeds the length that an int can handle
***/
bool parse_number(const std::string& key_string, const std::string& delimiter, int& number) ;


bool parse_batch_id(const std::string& key_string, int& client_id, int& batch_id);


bool parse_query_info(const std::string& key_string, int& client_id, int& batch_id, int& cluster_id, int& qid);

struct CompareObjKey {
     bool operator()(const std::string& key1, const std::string& key2) const {
          size_t pos = key1.rfind("/");
          int num1 = std::stoi(key1.substr(pos + 1));
          pos = key2.rfind("/");
          int num2 = std::stoi(key2.substr(pos + 1));
          return num1 > num2;
     }
};

/*** Helper function to callers of list_key:
*    filter keys that doesn't have exact prefix, or duplicate keys (from experiment at scale, it occurs.)
*    e.g. /doc1/1, /doc12/1, which return by list_keys("/doc1"), but are not for the same cluster
*    TODO: adjust op_list_keys semantics? 
*/
std::priority_queue<std::string, std::vector<std::string>, CompareObjKey> filter_exact_matched_keys(std::vector<std::string>& obj_keys, const std::string& prefix);

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
                                                            std::vector<std::string>& query_list);

void new_deserialize_embeddings_and_quries_from_bytes(const uint8_t* bytes,
                                                            const std::size_t& data_size,
                                                            uint32_t& nq,
                                                            const int& emb_dim,
                                                            float*& query_embeddings,
                                                            std::vector<std::string>& query_list);

/***
* Format the search results for each query to send to the next UDL.
* The format is | top_k | embeding_id_vector | distance_vector | query_text |
***/
std::string serialize_cluster_search_result(uint32_t top_k, long* I, float* D, int idx, const std::string& query_text);


struct DocIndex{
     int cluster_id;
     long emb_id;
     float distance;
     bool operator<(const DocIndex& other) const {
          return distance < other.distance;
     }
};

inline std::ostream& operator<<(std::ostream& os, const DocIndex& doc_index) {
     os << "cluster_id: " << doc_index.cluster_id << ", emb_id: " << doc_index.emb_id << ", distance: " << doc_index.distance;
     return os;
}


/***
 * Helper function to aggregate cdpo_handler()
 * 
***/
void deserialize_cluster_search_result_from_bytes(const int& cluster_id,
                                                  const uint8_t* bytes,
                                                  const size_t& data_size,
                                                  std::string& query_text,
                                                  std::vector<DocIndex>& cluster_results);

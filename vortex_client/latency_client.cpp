#include <cascade/service_client_api.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <unistd.h>  
#include <vector>

using namespace derecho::cascade;
#define EMBEDDING_DIM 1024
#define VORTEX_SUBGROUP_INDEX 0
#define AGG_SUBGROUP_INDEX 0

// Use vector since one query may be reuse for multiple times
std::unordered_map<std::string, std::vector<std::tuple<int, int>>> sent_queries;
std::unordered_map<std::string, std::string> query_results;

std::vector<std::string> read_queries(std::string query_director) {
     std::vector<std::string> queries;
     std::ifstream file(query_director);
     if (!file.is_open()) {
          std::cerr << "Error: Could not open query directory:" << query_director << std::endl;
          std::cerr << "Current only support query_doc in csv format." << std::endl;
          return queries;
     }
     std::string line;
     while (std::getline(file, line)) {
          queries.push_back(line);
     }
     file.close();
     return queries;
}

std::mt19937 rng;
/***
 * Generate randomly temporarily, 
 * TODO: use OpenAI API call to get the embeddings
 * @param d the dimension of the embeddings
 * @param nq the number of queries
 */
bool generate_embeddings(int d, int nq, std::unique_ptr<float[]>& xq){
     std::uniform_real_distribution<> distrib;
     xq = std::make_unique<float[]>(d * nq);
     for (int i = 0; i < nq; i++) {
          for (int j = 0; j < d; j++)
               xq[d * i + j] = distrib(rng);
          xq[d * i] += i / 1000.;
     }
     return true;
}


std::string format_query_emb_object(int nq, std::unique_ptr<float[]>& xq, std::vector<std::string>& query_list) {
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
                              std::string(reinterpret_cast<const char*>(query_embeddings), sizeof(float) * EMBEDDING_DIM * num_queries) +
                              nlohmann::json(query_list).dump();
     return query_emb_string;
}

/***
 * Result JSON is in format of : {"query": query_text, "top_k_docs":[doc_text1, doc_text2, ...]}
 */
bool deserialize_result(const Blob& blob, std::string& query_text, std::vector<std::string>& top_k_docs) {
     if (blob.size == 0) {
          std::cerr << "Error: empty result blob." << std::endl;
          return false;
     }
     char* json_data = const_cast<char*>(reinterpret_cast<const char*>(blob.bytes));
     std::size_t json_size = blob.size;
     std::string json_string(json_data, json_size);
     try{
          nlohmann::json parsed_json = nlohmann::json::parse(json_string);
          if (parsed_json.count("query") == 0 || parsed_json.count("top_k_docs") == 0) {
               std::cerr << "Result JSON does not contain query or top_k_docs." << std::endl;
               return false;
          }
          query_text = parsed_json["query"];
          top_k_docs = parsed_json["top_k_docs"];
     } catch (const nlohmann::json::parse_error& e) {
          std::cerr << "Result JSON parse error: " << e.what() << std::endl;
          return false;
     }
     return true;
}

bool run_latency_test(ServiceClientAPI& capi, int num_queries, int batch_size, std::string& query_directory, int query_interval) {
     int my_id = capi.get_my_id();
     // 1.1. Prepare for the notification by creating object pool for results to store
     std::string result_pool_name = "/rag/results/" + std::to_string(my_id);
     auto res = capi.template create_object_pool<VolatileCascadeStoreWithStringKey>(result_pool_name,0,HASH,{});
     for (auto& reply_future:res.get()) {
          reply_future.second.get(); // wait for the object pool to be created
     }
     std::cout << "Created object pool for results: " << result_pool_name << std::endl;
     // 1.2. Register notification for this object pool
     std::atomic<bool> running(true);
     bool ret = capi.register_notification_handler(
               [&](const Blob& result){
                    // std::cout << "Subgroup Notification received:"
                    //       << "data:" << std::string(reinterpret_cast<const char*>(result.bytes),result.size)
                    //       << std::endl;
                    std::string query_text;
                    std::vector<std::string> top_k_docs;
                    if (!deserialize_result(result, query_text, top_k_docs)) {
                         std::cerr << "Error: failed to deserialize the result from the notification." << std::endl;
                         return false;
                    }
                    if (query_results.find(query_text) == query_results.end()) {
                         query_results[query_text] = top_k_docs[0];
                    }
                    if (sent_queries.find(query_text) != sent_queries.end()) {
                         for (auto& [qb_id, q_id]: sent_queries[query_text]) {
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
                              TimestampLogger::log(LOG_TAG_QUERIES_RESULT_CLIENT_RECEIVED,my_id,qb_id,q_id);
#endif
                              std::cout << "Received result for query: " << query_text << " from client: " << my_id << " qb_id: " << qb_id << " q_id: " << q_id << std::endl;
                              // remove [qb_id, q_id] from sent_queries
                              sent_queries[query_text].erase(std::remove(sent_queries[query_text].begin(), sent_queries[query_text].end(), std::make_tuple(qb_id, q_id)), sent_queries[query_text].end());
                              if (sent_queries[query_text].size() == 0) {
                                   sent_queries.erase(query_text);
                              }
                              break;
                         }
                    } else {
                         std::cerr << "Error: received result for query that is not sent." << std::endl;
                    }
                    if (sent_queries.size() == 0) {
                         running = false;
                         std::cout << "Received all results." << std::endl;
                    }
                    return true;
               }, result_pool_name);
     if (ret) {
          std::cerr << "Replaced previous notification handler for obj_pool: " << result_pool_name << std::endl;
     }

     // 1.3. Establishing connections to all server nodes
     auto subgroup_members = capi.template get_subgroup_members<VolatileCascadeStoreWithStringKey>(AGG_SUBGROUP_INDEX);
     int num_shards = subgroup_members.size();
     std::string establish_key = "establish_connection";
     for (int i = 0; i < num_shards; i++) {
          ObjectWithStringKey obj;
          std::string control_value = "establish";
          obj.key = establish_key;
          obj.blob = Blob(reinterpret_cast<const uint8_t*>(control_value.c_str()), control_value.size());
          auto res = capi.template put<VolatileCascadeStoreWithStringKey>(obj, AGG_SUBGROUP_INDEX, i, true);
          for (auto& reply_future:res.get()) {
               reply_future.second.get(); // wait for the object has been put
          }
     }


     // 2. Prepare the query
     std::vector<std::string> queries = read_queries(query_directory);
     size_t tatal_q_num = queries.size();
     if (queries.size() == 0) {
          std::cerr << "Error: failed to read queries from " << query_directory << std::endl;
          return false;
     }

     // 3. send the queries to the cascade
     for (int qb_id = 0; qb_id < num_queries; qb_id++) {
          std::string key = "/rag/emb/centroids_search/client" + std::to_string(my_id) + "/qb" + std::to_string(qb_id);
          // 3.1. Prepare the query texts
          std::vector<std::string> cur_query_list;
          for (int j = 0; j < batch_size; ++j) {
              cur_query_list.push_back(queries[(qb_id + j) % tatal_q_num]);
              if (sent_queries.find(queries[(qb_id + j) % tatal_q_num]) == sent_queries.end()) {
                   sent_queries[queries[(qb_id + j) % tatal_q_num]] = std::vector<std::tuple<int, int>>();
              }
              sent_queries[queries[(qb_id + j) % tatal_q_num]].push_back(std::make_tuple(qb_id, j));
          }
          // 3.2. Generate embedding for the query
          std::unique_ptr<float[]> query_embeddings;
          generate_embeddings(EMBEDDING_DIM, batch_size, query_embeddings);
          // 3.3. format the query 
          std::string emb_query_string = format_query_emb_object(batch_size, query_embeddings, cur_query_list);
          ObjectWithStringKey emb_query_obj;
          emb_query_obj.key = key;
          emb_query_obj.blob = Blob(reinterpret_cast<const uint8_t*>(emb_query_string.c_str()), emb_query_string.size());
          // 3.4. send the object to the cascade
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
          for (int j = 0; j < batch_size; ++j) {
               TimestampLogger::log(LOG_TAG_QUERIES_SENDING_START,my_id,qb_id,j);
          }
#endif
          capi.put_and_forget(emb_query_obj, false); // not only trigger the UDL, but also update state. TODO: Need more thinking here. 
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
          for (int j = 0; j < batch_size; ++j) {
               TimestampLogger::log(LOG_TAG_QUERIES_SENDING_END,my_id,qb_id,j);
          }
#endif
          // std::this_thread::sleep_for(std::chrono::microseconds(query_interval));
          // implement sleep using busy waiting and while loop
          auto start = std::chrono::high_resolution_clock::now();
          while (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() < query_interval) {
               // busy waiting
          }
     }
     
     std::cout << "Put all queries to cascade." << std::endl;
     // std::this_thread::sleep_for(std::chrono::seconds(10));
     // running = false;
     // message_thread.join();
     while (running.load()) {
          // std::this_thread::sleep_for(std::chrono::seconds(1));
     }

     // 4. flush logs
     std::string flush_log_key = "/rag/emb/centroids_search/flush_logs";
     for (int i = 0; i < num_shards; i++){
          ObjectWithStringKey obj;
          std::string control_value = "flush";
          obj.key = flush_log_key;
          obj.blob = Blob(reinterpret_cast<const uint8_t*>(control_value.c_str()), control_value.size());
          auto res = capi.template put<VolatileCascadeStoreWithStringKey>(obj, VORTEX_SUBGROUP_INDEX, i, true);
          for (auto& reply_future:res.get()) {
               reply_future.second.get(); // wait for the object pool to be created
          }
     }
     std::cout << "Flushed logs to shards." << std::endl;
     TimestampLogger::flush("client_timestamp.dat");
     return true;
}


int main(int argc, char** argv){
     int opt;
     int num_queries = 0;
     int batch_size = 0;
     // int query_interval = 100000; // default interval between query is 1 second
     int query_interval = 50000;
     std::string query_directory = "";

     while ((opt = getopt(argc, argv, "n:b:q:")) != -1) {
          switch (opt) {
               case 'n':
                    num_queries = std::atoi(optarg);  // Convert the argument to an integer
                    break;
               case 'b':
                    batch_size = std::atoi(optarg);   // Convert the argument to an integer
                    break;
               case 'q':
                    query_directory = optarg;
                    break;
               case 'i':
                    query_interval = std::atoi(optarg);
                    break;
               case '?': // Unknown option or missing option argument
                    std::cerr << "Usage: " << argv[0] << " -n <number_of_queries> -b <batch_size> -q <query_dir.csv> -i <interval>" << std::endl;
                    return 1;
               default:
                    break;
          }
     }
     if (num_queries == 0 || batch_size == 0 || query_directory.empty()) {
          std::cerr << "Error: Missing required options." << std::endl;
          std::cerr << "Usage: " << argv[0] << " -n <number_of_queries> -b <batch_size> -q <query_dir.csv>" << std::endl;
          return 1;
     }

     std::cout << "Number of queries: " << num_queries << std::endl;
     std::cout << "Batch size: " << batch_size << std::endl;

     rng.seed(42);
     auto& capi = ServiceClientAPI::get_service_client();
     run_latency_test(capi, num_queries, batch_size, query_directory, query_interval);

     return 0;
}

#include "vortex_client.hpp"

using namespace derecho::cascade;
// maximum number of embeddings could be batched per object, based on emb_dim and p2p message size in derecho.cfg
#define MAX_NUM_EMB_PER_OBJ 200  

int main(int argc, char** argv){
     int opt;
     int num_batches = 0;
     int batch_size = 0;
     // int query_interval = 100000; // default interval between query is 1 second
     int query_interval = 50000;
     int emb_dim = 1024;
     std::string query_directory = "";
     bool only_send_query_text = false;

     while ((opt = getopt(argc, argv, "n:b:q:i:e:t")) != -1) {
          switch (opt) {
               case 'n':
                    num_batches = std::atoi(optarg);  // Convert the argument to an integer
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
               case 'e':
                    emb_dim = std::atoi(optarg);
                    break;
               case 't':
                    only_send_query_text = true;
                    break;
               case '?': // Unknown option or missing option argument
                    std::cerr << "Usage: " << argv[0] << " -n <number_of_batches> -b <batch_size> -q <query_data_dir> -i <interval> -e <emb_dim> [-t] (if only send query text)" << std::endl;
                    return 1;
               default:
                    break;
          }
     }
     if (num_batches == 0 || batch_size == 0 || query_directory.empty()) {
          std::cerr << "Error: Missing required options." << std::endl;
          std::cerr << "Usage: " << argv[0] << " -n <number_of_batches> -b <batch_size> -q <query_dir.csv> -i <interval> -e <emb_dim> [-t] (if only send query text)" << std::endl;
          return 1;
     }
     if (batch_size > MAX_NUM_EMB_PER_OBJ) {
          std::cerr << "Error: batch_size="<< batch_size << " exceeds MAX_NUM_EMB_PER_OBJ=" << MAX_NUM_EMB_PER_OBJ << "." << std::endl;
          return 1;
     }

     std::cout << "Number of batches: " << num_batches << std::endl;
     std::cout << "Batch size: " << batch_size << std::endl;

     auto& capi = ServiceClientAPI::get_service_client();
     int node_id = capi.get_my_id();
     VortexPerfClient perf_client(node_id, num_batches, batch_size, query_interval, emb_dim, only_send_query_text);

     // 1. Register notification on all servers
     int num_shards = perf_client.register_notification_on_all_servers(capi);
     if (num_shards == -1) {
          std::cerr << "Error: failed to establish connections to all servers." << std::endl;
          return 1;
     }
     // 2. Prepare the query and query embeddings
     std::vector<std::string> queries;
     std::unique_ptr<float[]> query_embs;
     bool prepare_success = perf_client.prepare_queries(query_directory, queries, query_embs);
     if (!prepare_success) {
          std::cerr << "Error: failed to prepare queries." << std::endl;
          return false;
     }
     // 3. Run perf test, sending queries and wait for results
     perf_client.run_perf_test(capi, queries, query_embs);

     // 4. Flush logs
     perf_client.flush_logs(capi, num_shards);

     perf_client.compute_recall(capi, query_directory);

     return 0;
}

#include "vortex_client.hpp"

using namespace derecho::cascade;
// maximum number of embeddings could be batched per object, based on emb_dim and p2p message size in derecho.cfg
#define MAX_NUM_EMB_PER_OBJ 200  

int main(int argc, char** argv){
     int opt;
     int num_queries = 0;
     int batch_size = 0;
     // int query_interval = 100000; // default interval between query is 1 second
     int query_interval = 50000;
     int emb_dim = 1024;
     std::string query_directory = "";

     while ((opt = getopt(argc, argv, "n:b:q:i:e:")) != -1) {
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
               case 'e':
                    emb_dim = std::atoi(optarg);
                    break;
               case '?': // Unknown option or missing option argument
                    std::cerr << "Usage: " << argv[0] << " -n <number_of_queries> -b <batch_size> -q <query_data_dir> -i <interval> -e <emb_dim>" << std::endl;
                    return 1;
               default:
                    break;
          }
     }
     if (num_queries == 0 || batch_size == 0 || query_directory.empty()) {
          std::cerr << "Error: Missing required options." << std::endl;
          std::cerr << "Usage: " << argv[0] << " -n <number_of_queries> -b <batch_size> -q <query_dir.csv> -i <interval> -e <emb_dim>" << std::endl;
          return 1;
     }
     if (batch_size > MAX_NUM_EMB_PER_OBJ) {
          std::cerr << "Error: batch_size="<< batch_size << " exceeds MAX_NUM_EMB_PER_OBJ=" << MAX_NUM_EMB_PER_OBJ << "." << std::endl;
          return 1;
     }

     std::cout << "Number of queries: " << num_queries << std::endl;
     std::cout << "Batch size: " << batch_size << std::endl;

     auto& capi = ServiceClientAPI::get_service_client();
     int node_id = capi.get_my_id();
     VortexPerfClient perf_client(node_id, num_queries, batch_size, query_interval, emb_dim);

     // 1. Register notification on all servers
     int num_shards = perf_client.register_notification_on_all_servers(capi);
     if (num_shards == -1) {
          std::cerr << "Error: failed to establish connections to all servers." << std::endl;
          return 1;
     }
     // 2. Run perf test
     perf_client.run_perf_test(capi, query_directory);

     // 3. Flush logs
     perf_client.flush_logs(capi, num_shards);

     perf_client.compute_recall(capi, query_directory);

     return 0;
}


#include "vortex_webservice.hpp"

void VortexWebService::run(){
    std::cout << "Starting Vortex Cascade client ..." << std::endl;
    VortexCascadeClient vortex;
    vortex.setup(batch_min_size,batch_max_size,batch_time_us,emb_dim,num_result_threads);

    // handler for HTTP requests
    request_handler_t request_handler = [&](http::request<http::string_body>&& req){
        uint64_t dim = req.body().size() / sizeof(float);

        // POST method
        if ((req.method() == http::verb::post) && (dim == emb_dim)) {
            // send query to vortex and wait for the result
            const float *emb = reinterpret_cast<const float *>(req.body().c_str());
            auto fut = vortex.query("",emb);

            if(fut.wait_for(std::chrono::seconds(VORTEX_MAX_WAIT_TIME)) != std::future_status::timeout){
                // build the response
                http::response<http::string_body> res{http::status::ok, req.version()};
                res.set(http::field::server, "Vortex");
                res.set(http::field::content_type, "text/plain");
                res.keep_alive(req.keep_alive());
                res.body() = boost::algorithm::join(fut.get(), "@");
                res.prepare_payload();
                return res;
            } else {
                std::cerr << "Timeout while waiting for future" << std::endl;
            }
        }

        // default response
        http::response<http::string_body> res{http::status::bad_request, req.version()};
        res.set(http::field::server, "Vortex");
        res.set(http::field::content_type, "text/plain");
        res.keep_alive(req.keep_alive());
        res.body() = "Method must be POST and embeddings dimension must be " + std::to_string(emb_dim) + " (received: " + std::to_string(dim) + ")";
        res.prepare_payload();
        return res;
    };
    
    std::cout << "Starting to listen to requests ..." << std::endl;
    auto const net_address = net::ip::make_address(address);
    try {
        net::io_context ioc{num_io_threads};
        auto listener = std::make_shared<Listener>(ioc, btcp::endpoint{net_address, port},request_handler);
        listener->run();

        // multi-threaded listening
        std::vector<std::thread> threads;
        threads.reserve(num_io_threads-1);

        for(auto i=num_io_threads-1; i>0; i--){
            threads.emplace_back([&ioc]{
                ioc.run();
            });
        }

        ioc.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}


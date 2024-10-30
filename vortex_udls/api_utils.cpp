#include "api_utils.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>

namespace api_utils {

// Callback function for handling curl response
size_t writeCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    ((std::string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
}

/*** Curl get raw embedding JSON response from OpenAI API 
 * @param query: Input text query
 * @param model: Model name for text embedding
 * @param api_key: OpenAI API key
 * @param dim: Dimension of the embedding
 * @return: Raw JSON response from OpenAI API
 */
std::string get_embedding(const std::string &query, const std::string &model, const std::string &api_key, int dim) {
    CURL *curl;
    CURLcode res;
    std::string readBuffer;

    curl = curl_easy_init();
    if (curl) {
        // Include dimensions in payload if your API allows it
        nlohmann::json payload = {
            {"model", model},
            {"input", query},
            {"dimensions", dim}  // Include dimensions as a parameter
        };

        std::string url = "https://api.openai.com/v1/embeddings";
        std::string postData = payload.dump();

        struct curl_slist *headers = nullptr;
        headers = curl_slist_append(headers, ("Authorization: Bearer " + api_key).c_str());
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postData.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
    }

    return readBuffer;  // Return raw JSON response from OpenAI
}

// Function to run GPT model with query and top documents
std::string run_gpt4o_mini(const std::string &query, const std::vector<std::string> &top_docs, const std::string &model, const std::string &api_key) {
    CURL *curl;
    CURLcode res;
    std::string readBuffer;

    curl = curl_easy_init();
    if (curl) {
        // Combine the query and top documents into a single conversation-like input
        std::string prompt = query + "\nTop documents:\n";
        for (const auto& doc : top_docs) {
            prompt += "- " + doc + "\n";
        }

        // Use the chat/completions endpoint with properly formatted messages
        nlohmann::json payload = {
            {"model", model},
            {"messages", {
                {{"role", "system"}, {"content", "You are a helpful assistant."}},
                {{"role", "user"}, {"content", prompt}}
            }}
        };

        std::string url = "https://api.openai.com/v1/chat/completions";
        std::string postData = payload.dump();

        struct curl_slist *headers = nullptr;
        headers = curl_slist_append(headers, ("Authorization: Bearer " + api_key).c_str());
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postData.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
    }

    // Parse the response to extract only the assistant's message content
    try {
        auto json_response = nlohmann::json::parse(readBuffer);
        if (json_response.contains("error")) {
            std::cerr << "API error: " << json_response["error"]["message"] << std::endl;
            return "Error: " + json_response["error"]["message"].get<std::string>();
        }
        std::string answer = json_response["choices"][0]["message"]["content"];
        return answer;
    } catch (const std::exception &e) {
        std::cerr << "Error parsing JSON response: " << e.what() << std::endl;
        return "Error: Unable to parse response.";
    }
}


}  // namespace api_utils
